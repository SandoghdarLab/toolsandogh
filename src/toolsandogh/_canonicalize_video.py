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
    # optional metadata
    acquisition_date: datetime | None = None,
    creator: str | None = None,
    dt: float | None = None,
    dz: float | None = None,
    dy: float | None = None,
    dx: float | None = None,
) -> xr.DataArray:
    """
    Turn the supplied data into its canonical TCZYX video representation.

    Parameters
    ----------
    video : xr.DataArray
        An xarray.
    acquisition_date : datetime.datetime
        A timestamp of when the video was created.  Defaults to datetime.now().
    creator : str
        A string describing who created the video.
    dt : float
        The time interval in milliseconds between one video frame and the next.
        Defaults to 1/60 of a second.
    dz : float
        The spatial distance in micrometer between one Z slice and the next.
        Defaults to 1 micrometer.
    dy : float
        The spatial distance in micrometer between one row of pixels and the next.
        Defaults to 1 micrometer.
    dx : float
        The spatial distance in micrometer between one column of pixels and the next.
        Defaults to 1 micrometer.

    Returns
    -------
    xarray.DataArray
        A TCZYX video with the supplied parameters.
    """
    # Turn video into an xarray.
    if not isinstance(video, xr.DataArray):
        video = video_from_array(video)

    # Ensure the video's data is a Dask array.
    if not isinstance(video.data, da.Array):
        video = video.copy(data=da.from_array(video.data), deep=False)

    # Ensure the TZYX axes exist and are continuous.
    for dim in ("T", "Z", "Y", "X"):
        if dim not in video.dims:
            video = video.expand_dims({dim: np.arange(1.0)})
        elif video[dim].dtype != np.float64:
            values = np.array(video[dim])
            video = video.assign_coords({dim: values.astype(np.float64)})

    # Ensure the C axis exists.
    if "C" not in video.dims:
        video = video.expand_dims({"C": 1})

    # If there is a S axis, merge its entries.
    if "S" in video.dims:
        video = video.mean("S", dtype=np.float32).astype(video.dtype)

    # Ensure the correct ordering of axes.
    video = video.transpose("T", "C", "Z", "Y", "X")

    # Ensure there is metadata attached.
    if not hasattr(video, "processed"):
        (size_t, size_c, size_z, size_y, size_x) = video.shape
        pixels = Pixels(
            type=dtype_pixel_type(video.dtype),
            big_endian=dtype_is_big_endian(video.dtype),
            dimension_order=Pixels_DimensionOrder.XYZCT,
            size_t=size_t,
            size_c=size_c,
            size_z=size_z,
            size_y=size_y,
            size_x=size_x,
            time_increment=dt or (1000 / 60),
            physical_size_z=dz or 1.0,
            physical_size_y=dy or 1.0,
            physical_size_x=dx or 1.0,
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
    # TODO: Update metadata with any supplied parameters.

    # Raise an exception if the video is still not in canonical form.
    validate_video(video)

    # Done.
    return video


def video_from_array(array: npt.ArrayLike) -> xr.DataArray:
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


def dtype_pixel_type(dtype: npt.DTypeLike) -> PixelType:
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


def dtype_is_big_endian(dtype: npt.DTypeLike) -> bool:
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
