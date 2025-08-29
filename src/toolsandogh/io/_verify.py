import dask.array as da
import numpy as np
import numpy.typing as npt
import xarray as xr


def verify_video(
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
    rtol=1e-3,
) -> None:
    """
    Assert that the supplied video obeys the following rules:

    1. It is of type xarray.DataArray.

    2. Its data is represented as dask.array.Array.

    3. Its dims are ('T', 'C', 'Z', 'Y', 'X').

    4. Any T, C, Z, Y, or X that is not None matches the size of the corresponding video axis.

    5. Any T, Z, Y, or X coordinate with size larger than one is of type np.float64.

    6. Any dt, dz, dy, or dx argument that is not None describes the step size of the corresponding coordinate.

    7. If the dtype argument is not None, it matches the video's dtype.
    """
    assert isinstance(video, xr.DataArray)
    assert isinstance(video.data, da.Array)
    dims = ("T", "C", "Z", "Y", "X")
    sizes = (T, C, Z, Y, X)
    steps = (dt, None, dz, dy, dx)
    assert video.dims == dims
    for dim, size, step in zip(dims, sizes, steps):
        coord = video[dim]
        if size is not None:
            assert len(coord) == size
        if len(coord) > 1 and step is not None:
            assert coord.dtype == np.float64
            array: npt.NDArray[np.float64] = coord.values
            delta = array[1] - array[0]
            error = abs(delta - step)
            assert (error / step) <= rtol
    if dtype is not None:
        assert video.dtype == dtype
