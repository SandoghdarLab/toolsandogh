import dask.array as da
import numpy as np
import numpy.typing as npt
import xarray as xr

from ._verify import verify_video


def generate_video(
    T: int = 10,
    C: int = 1,
    Z: int = 1,
    Y: int = 128,
    X: int = 128,
    dt: float = 1.0,
    dz: float = 1.0,
    dy: float = 1.0,
    dx: float = 1.0,
    dtype: npt.DTypeLike = np.float32,
) -> xr.DataArray:
    """
    Generate a microscopy dataset filled with random noise.

    Args:
        T (int): The extent of the time axis.
        C (int): The number of channels.
        Z (int): The number of Z slices.
        Y (int): The number of rows.
        X (int): The number of columns.
        dt (float): The step size in the T (time) dimension.
        dz (float): The step size in the Z (height) dimension.
        dy (float): The step size in the Y dimension.
        dx (float): The step size in the X dimension.
        dtype (npt.DTypeLine): The Numpy dtype of the resulting dataset.

    Returns:
        xarray.DataArray: A random TCZYX dataset with the specified parameters.
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
    verify_video(
        video, T=T, C=C, Z=Z, Y=Y, X=X, dt=dt, dz=dz, dy=dy, dx=dx, dtype=dtype
    )
    return video
