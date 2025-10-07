import dask.array as da
import numpy as np
import numpy.typing as npt
import xarray as xr

from ._canonicalize_video import canonicalize_video


def generate_video(
    T: int = 1,
    C: int = 1,
    Z: int = 1,
    Y: int = 1,
    X: int = 1,
    dt: float = 1.0,
    dz: float = 1.0,
    dy: float = 1.0,
    dx: float = 1.0,
    dtype: npt.DTypeLike = np.float32,
) -> xr.DataArray:
    """
    Generate a video filled with random noise.

    Parameters
    ----------
    T : int
        The T (time) extent of the video.
    C : int
        The C (channel) extent of the video.
    Z : int
        The Z (height) extent of the video.
    Y : int
        The Y (row) extent of the video.
    X : int
        The X (column) extent of the video.
    dt : float
        The T (time) step size of the video.
    dz : float
        The Z (height) step size of the video.
    dy : float
        The Y (row) step size of the video.
    dx : float
        The X (column) step size of the video.
    dtype : npt.DtypeLike
        The dtype of the resulting video.

    Returns
    -------
    xarray.DataArray
        A TCZYX video with the supplied parameters.
    """
    rng = da.random.default_rng()
    video = canonicalize_video(
        rng.random(
            size=(T, C, Z, Y, X),
            dtype=dtype,  # type: ignore
        ),
    )

    # Check that the video matches the supplied parameters.
    for dim, size, step in zip("TCZYX", (T, C, Z, Y, X), (dt, None, dz, dy, dx)):
        coord = video[dim]
        assert len(coord) == size
        if len(coord) > 1 and step is not None:
            array = coord.values
            delta = array[1] - array[0]
            error = abs(delta - step)
            assert (error / step) <= 1e-3
    assert video.dtype == dtype

    # Done.
    return video
