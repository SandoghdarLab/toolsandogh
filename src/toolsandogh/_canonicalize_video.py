import sys
from datetime import datetime, timezone

import dask.array as da
import numpy as np
import numpy.typing as npt
import xarray as xr
from ome_types.model import (
    OME,
    Image,
    Pixels,
    Pixels_DimensionOrder,
    PixelType,
    UnitsLength,
    UnitsTime,
)

from ._validate_video import validate_video


def canonicalize_video(
    video: npt.ArrayLike,
    # optional video properties
    T: int | None = None,
    C: int | None = None,
    Z: int | None = None,
    Y: int | None = None,
    X: int | None = None,
    dt: float | None = None,
    dz: float | None = None,
    dy: float | None = None,
    dx: float | None = None,
    t0: float | None = None,
    z0: float | None = None,
    y0: float | None = None,
    x0: float | None = None,
    dtype: npt.DTypeLike | None = None,
    # optional video metadata
    acquisition_date: datetime | None = None,
    creator: str | None = None,
) -> xr.DataArray:
    """
    Turn the supplied data into its canonical TCZYX video representation.

    Parameters
    ----------
    video : npt.ArrayLike
        An array-like entity with up to five axes.  If the axes are not labeled
        already, a 1D array is treated as having an `X` axis only, a 2D array is
        treated as having dimensions `XY`, a 3D array is treated as having
        dimensions `TYX`, a 4D array is treated as having dimensions `TZYX`, and
        a 5D array is treated as having dimensions `TCZYX`.
    T : int
        The expected T (time) extent of the video.
    C : int
        The expected C (channel) extent of the video.
    Z : int
        The expected Z (height) extent of the video.
    Y : int
        The expected Y (row) extent of the video.
    X : int
        The expected X (column) extent of the video.
    dt : float, optional
        The time interval in milliseconds between one video frame and the
        next.  When ``None`` (the default) the spacing is inferred from the
        video's ``T`` coordinate when one is present, or else falls back to
        ``1000 / 60`` ms (60 fps).  If supplied together with an existing
        ``T`` coordinate, the two must agree to within a small tolerance.
    dz : float, optional
        The spatial distance in micrometre between one Z slice and the next.
        When ``None`` (the default) the spacing is inferred from the video's
        ``Z`` coordinate, or else defaults to 1 µm.  If supplied together
        with an existing ``Z`` coordinate, the two must agree.
    dy : float, optional
        The spatial distance in micrometre between one row of pixels and the
        next.  When ``None`` (the default) the spacing is inferred from the
        video's ``Y`` coordinate, or else defaults to 1 µm.  If supplied
        together with an existing ``Y`` coordinate, the two must agree.
    dx : float, optional
        The spatial distance in micrometre between one column of pixels and
        the next.  When ``None`` (the default) the spacing is inferred from
        the video's ``X`` coordinate, or else defaults to 1 µm.  If supplied
        together with an existing ``X`` coordinate, the two must agree.
    t0 : float, optional
        The time of the first frame, in milliseconds.  When ``None`` (the
        default) the origin is inferred from the video's ``T`` coordinate,
        or else defaults to zero.  If supplied together with an existing
        ``T`` coordinate, the two must agree.
    z0 : float, optional
        The position of the first Z slice, in micrometre.  When ``None``
        (the default) the origin is inferred from the video's ``Z``
        coordinate, or else defaults to zero.  If supplied together with an
        existing ``Z`` coordinate, the two must agree.
    y0 : float, optional
        The position of the first row of pixels, in micrometre.  When
        ``None`` (the default) the origin is inferred from the video's ``Y``
        coordinate, or else defaults to zero.  If supplied together with an
        existing ``Y`` coordinate, the two must agree.
    x0 : float, optional
        The position of the first column of pixels, in micrometre.  When
        ``None`` (the default) the origin is inferred from the video's ``X``
        coordinate, or else defaults to zero.  If supplied together with an
        existing ``X`` coordinate, the two must agree.
    dtype : npt.DTypeLike
        The expected dtype of the video.
    acquisition_date : datetime.datetime
        A timestamp of when the video was created.  Defaults to datetime.now().
    creator : str
        A string describing who created the video.

    Returns
    -------
    xarray.DataArray
        A TCZYX video with the supplied parameters that passes :py:func:`~toolsandogh.validate_video`.

    Notes
    -----
    The ``T``, ``Z``, ``Y`` and ``X`` axes carry continuous, uniformly
    spaced coordinates in milliseconds (time) and micrometres (space),
    with the ``units`` attribute set on each.  For each axis the scale
    (``dt``, ``dz``, ``dy``, ``dx``) and origin (``t0``, ``z0``, ``y0``,
    ``x0``) are resolved by one of three routes: taken from an existing
    coordinate on the video (e.g. physical coordinates supplied by a
    reader), taken from the explicit parameter, or filled with a
    default.  When a parameter is supplied *and* the video already
    carries a coordinate, the two must agree to within a small
    tolerance; the coordinate values are always retained.  The channel
    (``C``) axis is categorical and carries no units.
    """
    # Turn video into an xarray.
    if not isinstance(video, xr.DataArray):
        video = _coerce_to_xarray(video)

    # Ensure the video's data is a Dask array.
    if not isinstance(video.data, da.Array):
        video = video.copy(data=da.from_array(video.data), deep=False)

    # Ensure the TZYX axes exist.  Missing axes are added as size-1
    # dimensions so that the coordinate-resolution pass below treats
    # them uniformly.
    for dim in ("T", "Z", "Y", "X"):
        if dim not in video.dims:
            video = video.expand_dims({dim: 1})

    # Resolve the per-axis scale, origin, and coordinate values.  For
    # each axis the scale and origin are either inferred from an
    # existing coordinate (in which case any explicitly supplied value
    # is validated against it), taken from the explicit parameter, or
    # filled with a default.  Time defaults to 60 fps (``1000 / 60`` ms);
    # space defaults to 1 µm; origins default to zero.
    axis_params = {
        "T": (dt, t0, 1000.0 / 60.0, "ms"),
        "Z": (dz, z0, 1.0, "µm"),
        "Y": (dy, y0, 1.0, "µm"),
        "X": (dx, x0, 1.0, "µm"),
    }
    resolved_scales: dict[str, float] = {}
    for dim in ("T", "Z", "Y", "X"):
        scale_param, origin_param, default_scale, units = axis_params[dim]
        vals, scale = _resolve_axis(video, dim, scale_param, origin_param, default_scale)
        coord = xr.DataArray(vals, dims=(dim,), attrs={"units": units})
        video = video.assign_coords({dim: coord})
        resolved_scales[dim] = scale

    time_increment = resolved_scales["T"]
    physical_size_z = resolved_scales["Z"]
    physical_size_y = resolved_scales["Y"]
    physical_size_x = resolved_scales["X"]

    # Ensure the C axis exists.  The channel axis is categorical and
    # carries no physical units.
    if "C" not in video.dims:
        video = video.expand_dims({"C": 1})

    # Derive the result dtype.
    if dtype is None:
        result_dtype = video.dtype
    else:
        result_dtype = np.dtype(dtype)

    # If there is a S axis, merge its entries.
    if "S" in video.dims:
        video = video.mean("S", dtype=np.float32)

    # Ensure the resulting video has the correct dtype.
    if video.dtype != result_dtype:
        video = video.astype(result_dtype)

    # Ensure the correct ordering of axes.
    axes = ("T", "C", "Z", "Y", "X")
    video = video.transpose(*axes)

    # Ensure each axis is as large as expected.
    expected_shape = (T, C, Z, Y, X)
    for axis, expected, actual in zip(axes, expected_shape, video.shape):
        if not (expected is None or expected == actual):
            raise RuntimeError(f"Expected {axis}-axis of size {expected}, but got {actual}")

    # Ensure there is metadata attached.
    if not hasattr(video, "processed"):
        (size_t, size_c, size_z, size_y, size_x) = video.shape
        pixels = Pixels(
            type=_dtype_pixel_type(video.dtype),
            big_endian=_dtype_is_big_endian(video.dtype),
            dimension_order=Pixels_DimensionOrder.XYZCT,
            size_t=size_t,
            size_c=size_c,
            size_z=size_z,
            size_y=size_y,
            size_x=size_x,
            time_increment=time_increment,
            physical_size_z=physical_size_z,
            physical_size_y=physical_size_y,
            physical_size_x=physical_size_x,
            time_increment_unit=UnitsTime.MILLISECOND,
            physical_size_z_unit=UnitsLength.MICROMETER,
            physical_size_y_unit=UnitsLength.MICROMETER,
            physical_size_x_unit=UnitsLength.MICROMETER,
        )
        image = Image(
            acquisition_date=(acquisition_date or datetime.now(timezone.utc)),
            description="Video with auto-generated metadata.",
            pixels=pixels,
        )
        ome = OME(
            images=[image],
            creator=(creator or "MPL Erlangen, Sandoghdar Division, toolsandogh"),
        )
        video = video.assign_attrs({"processed": ome})

    # Raise an exception if the video is still not in canonical form.
    validate_video(video)

    # Done.
    return video


def _coerce_to_xarray(array: npt.ArrayLike) -> xr.DataArray:
    """
    Turn a supplied array into an xarray.

    Parameters
    ----------
    array : npt.ArrayLike
        An object designating an array.

    Returns
    -------
    xr.DataArray
        An xarray with the same content and dtype as the supplied array.
    """
    # Determine the Dask array holding the video's data.
    if isinstance(array, da.Array):
        data = array
    else:
        data = da.from_array(array)

    # Determine the appropriate dims
    rank = len(data.shape)
    match rank:
        case 0:
            dims = ()
        case 1:
            dims = ("X",)
        case 2:
            dims = ("Y", "X")
        case 3:
            dims = ("T", "Y", "X")
        case 4:
            dims = ("T", "Z", "Y", "X")
        case 5:
            dims = ("T", "C", "Z", "Y", "X")
        case 6:
            dims = ("T", "C", "Z", "Y", "X", "S")
        case _:
            raise RuntimeError(f"Cannot interpret {rank}-dimensional data as a video.")

    # Create the xarray.
    return xr.DataArray(data=data, dims=dims)


def _resolve_axis(
    video: xr.DataArray,
    dim: str,
    scale: float | None,
    origin: float | None,
    default_scale: float,
) -> tuple[np.ndarray, float]:
    """
    Resolve the coordinate array and physical scale for one TZYX axis.

    When the video already carries a coordinate for ``dim``, its values
    are retained and the scale and origin are inferred from them; any
    explicitly supplied ``scale`` or ``origin`` is checked against the
    inferred values and must agree to within a small tolerance, otherwise
    a :class:`ValueError` is raised.

    When the video does not carry a coordinate, one is generated from the
    supplied scale and origin (falling back to ``default_scale`` and
    ``0.0`` respectively).

    Returns the coordinate values (as ``float64``) and the resolved
    physical scale (for the OME metadata).
    """
    size = int(video.sizes[dim])
    if dim in video.coords:
        vals = np.asarray(video[dim].values, dtype=np.float64)
        inferred_origin = float(vals[0])
        if origin is not None and not np.isclose(origin, inferred_origin, rtol=1e-6, atol=1e-9):
            raise ValueError(
                f"The supplied {dim.lower()}0={origin!r} does not match the "
                f"{dim} coordinate origin {inferred_origin!r}."
            )
        if size >= 2:
            inferred_scale = float(vals[1] - vals[0])
            if scale is not None and not np.isclose(scale, inferred_scale, rtol=1e-6, atol=1e-9):
                raise ValueError(
                    f"The supplied d{dim.lower()}={scale!r} does not match the "
                    f"{dim} coordinate spacing {inferred_scale!r}."
                )
            return vals, inferred_scale
        # A size-1 axis has no spacing to infer.
        return vals, (scale if scale is not None else default_scale)
    # No existing coordinate: generate one.
    resolved_scale = scale if scale is not None else default_scale
    resolved_origin = origin if origin is not None else 0.0
    vals = resolved_origin + np.arange(size, dtype=np.float64) * resolved_scale
    return vals, resolved_scale


def _dtype_pixel_type(dtype: npt.DTypeLike) -> PixelType:
    """
    Return the OME Pixel type corresponding to the supplied dtype.

    Parameters
    ----------
    dtype : npt.DTypeLike
        The Numpy dtype to be used for representing pixel data.

    Returns
    -------
    ome_types.model.PixelType
        A suitable OME pixel type.
    """
    match np.dtype(dtype):
        case np.int8:
            return PixelType.INT8
        case np.int16:
            return PixelType.INT16
        case np.int32:
            return PixelType.INT32
        case np.uint8:
            return PixelType.UINT8
        case np.uint16:
            return PixelType.UINT16
        case np.uint32:
            return PixelType.UINT32
        case np.float32:
            return PixelType.FLOAT
        case np.float64:
            return PixelType.DOUBLE
        case np.complex64:
            return PixelType.COMPLEXFLOAT
        case np.complex128:
            return PixelType.COMPLEXDOUBLE
        # The dtypes np.int64 and np.uint64 have no OME equivalent.
        case np.int64:
            return PixelType.BIT
        case np.uint64:
            return PixelType.BIT
        case _:
            raise RuntimeError(f"Cannot interpret {dtype} as a OME pixel type.")


def _dtype_is_big_endian(dtype: npt.DTypeLike) -> bool:
    """
    Return whether the video's data is stored in big endian byte order.

    Parameters
    ----------
    dtype : npt.DTypeLike
        The Numpy dtype to be used for representing pixel data.

    Returns
    -------
    bool
        True when the dtype is big endian, False otherwise.
    """
    dtype = np.dtype(dtype)

    if dtype.itemsize == 1:
        return False

    match np.dtype(dtype).byteorder:
        case ">":
            return True
        case "<":
            return False
        case "=":
            return sys.byteorder == "big"
        case _:
            raise RuntimeError(f"Cannot determine endianness of {dtype}.")
