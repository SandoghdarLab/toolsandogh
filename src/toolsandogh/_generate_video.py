import dask.array as da
import numpy as np
import numpy.typing as npt
import xarray as xr

from ._validate_video import validate_video


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
    shape = (T, C, Z, Y, X)
    rng = da.random.default_rng()
    video = xr.DataArray(
        rng.random(size=shape, dtype=dtype),  # type: ignore
        coords={
            "T": dt * np.arange(T),
            "C": range(C),
            "Z": dz * np.arange(Z),
            "Y": dy * np.arange(Y),
            "X": dx * np.arange(X),
        },
    )
    validate_video(video, T=T, C=C, Z=Z, Y=Y, X=X, dt=dt, dz=dz, dy=dy, dx=dx, dtype=dtype)
    return video
