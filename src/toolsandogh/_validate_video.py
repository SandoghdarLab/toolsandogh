"""Define a helper function for ensuring that videos have a canonical representation."""

import numpy as np
import numpy.typing as npt
import xarray as xr


def validate_video(
    video,
    *,
    T: int | None = None,
    C: int | None = None,
    Z: int | None = None,
    Y: int | None = None,
    X: int | None = None,
    dt: float | None = None,
    dz: float | None = None,
    dy: float | None = None,
    dx: float | None = None,
    dtype: npt.DTypeLike | None = None,
    rtol: float = 1e-3,
) -> None:
    """
    Ensure the supplied video is an xarray with axes T, C, Z, Y, and X.

    More precisely, raise an exception unless the supplied video adheres to the
    following rules:

    1. It is of type xarray.DataArray.

    2. Its dims are ('T', 'C', 'Z', 'Y', 'X').

    3. Any T, C, Z, Y, or X that is not None matches the size of the corresponding video axis.

    4. Any T, Z, Y, or X coordinate with size larger than one is of type np.float64.

    5. Any dt, dz, dy, or dx argument that is not None describes the step size of the corresponding coordinate.

    6. If the dtype argument is not None, it matches the video's dtype.

    Parameters
    ----------
    video : Any
        The video to be validated.
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
    dt : float
        The expected T (time) step size of the video.
    dz : float
        The expected Z (height) step size of the video.
    dy : float
        The expected Y (row) step size of the video.
    dx : float
        The expected X (column) step size of the video.
    dtype : npt.DtypeLike
        The expected dtype of the video.
    rtol : float
        The maximum admissible relative difference between any expected step size and the actual step size.
    """
    if not isinstance(video, xr.DataArray):
        raise TypeError(f"Not an xarray.DataArray: {video}")
    dims = ("T", "C", "Z", "Y", "X")
    sizes = (T, C, Z, Y, X)
    steps = (dt, None, dz, dy, dx)
    if not video.dims == dims:
        raise ValueError(f"Dimension mismatch.  Expected {dims}, got {video.dims}.")

    # Ensure temporal and spatial coordinates are continuous.
    for dim in ("T", "Z", "Y", "X"):
        coord = video[dim]
        if coord.dtype != np.float64:
            raise TypeError(f"The {dim} coordinate is not continuous.")

    # Ensure each axis matches its expected size and step.
    for dim, size, step in zip(dims, sizes, steps, strict=False):
        coord = video[dim]
        if size is not None and len(coord) != size:
            raise ValueError(f"Expected dim {dim} to have size {size}, got {len(coord)}.")
        if len(coord) > 1 and step is not None:
            array = coord.values
            delta = array[1] - array[0]
            error = abs(delta - step)
            if not (error / step) <= rtol:
                raise ValueError(f"Expected dim {dim} to have step size {step}, got {delta}.")

    # Ensure the video has the expected dtype.
    if dtype is not None and video.dtype != dtype:
        raise ValueError(f"Expected video dtype {dtype}, got {video.dtype}.")
