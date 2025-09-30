"""Define a function for loading videos from a wide range of sources."""

import os
import os.path
import pathlib
import urllib.parse

import bioio
import bioio_czi
import bioio_nd2
import dask.array as da
import fsspec
import numpy as np
import numpy.typing as npt
import xarray as xr
from fsspec.utils import math

from ._canonicalize_video import canonicalize_video


def load_video(
    path: str | os.PathLike,
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
) -> xr.DataArray:
    """
    Load a video from the given path.

    The resulting video is a 5D xarray with dimensions TCZYX.  If any keyword
    arguments are supplied, this function asserts that the resulting video
    matches these arguments, e.g., Z=1 may be used assert that the result is a
    2D video.

    The suffix of the supplied path determines the method that is used for
    reading the file.  For .raw and .bin files, the X, Y arguments are
    mandatory, the number of frames T is inferred automatically, and the
    remaining arguments default to Z=1, C=1, dz=1.0, dy=dx=1.0, dt=1.0, and
    dtype=uint8.

    Parameters
    ----------
    path : os.PathLike
        The name of a file, a URI, or a path.
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

    Returns
    -------
    xarray.DataArray
        A canonical TCZYX array that matches the supplied parameters.
    """
    # Parse the supplied path.
    pathstr = str(path)
    url = urllib.parse.urlparse(pathstr)
    path = pathlib.Path(url.path)

    # Gather all keyword arguments for those load functions that require them.
    kwargs = {
        "T": T,
        "C": C,
        "Z": Z,
        "Y": Y,
        "X": X,
        "dt": dt,
        "dz": dz,
        "dy": dy,
        "dx": dx,
        "dtype": dtype,
    }

    # Load the video.
    protocols = fsspec.available_protocols()
    match (url.scheme or "file", path.suffix):
        case (scheme, ".bin" | ".raw") if scheme in protocols:
            video = load_raw_video(path, **kwargs)
        case (_, ".czi"):
            img = bioio.BioImage(pathstr, reader=bioio_czi.Reader)
            video = img.xarray_dask_data
        case (_, ".nd2"):
            img = bioio.BioImage(pathstr, reader=bioio_nd2.Reader)
            video = img.xarray_dask_data
        case (_, _):
            # Let bioio figure out the rest or raise an exception
            img = bioio.BioImage(pathstr)
            video = img.xarray_dask_data

    # Ensure the video is canonical and return.
    return canonicalize_video(video)


def load_raw_video(
    path: os.PathLike,
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
) -> xr.DataArray:
    """
    Load a video from the specified raw file.

    Parameters
    ----------
    path : os.PathLike
        The name of a file, a URI, or a path.
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

    Returns
    -------
    xarray.DataArray
        A canonical TCZYX array that matches the supplied parameters.
    """
    # The parameters X and Y are mandatory for loading raw files.
    if Y is None:
        raise RuntimeError("The parameter Y is required for loading raw files.")
    if X is None:
        raise RuntimeError("The parameter X is required for loading raw files.")
    # Z and C default to one.
    if Z is None:
        Z = 1
    if C is None:
        C = 1
    # The default dtype is np.uint8
    if dtype is None:
        dtype = np.uint8
    # The number of frames T can be inferred from the file's size.
    bits_per_item = 12 if (dtype == "uint12") else (np.dtype(dtype).itemsize * 8)
    nbits = os.path.getsize(path) * 8
    nitems = nbits // bits_per_item
    items_per_T = C * Z * Y * X
    if T is None:
        T = nitems // items_per_T
    nexpected = T * items_per_T
    if nexpected > nitems:
        raise RuntimeError(
            f"The file {path} has only {nitems} items, but {nexpected} were expected."
        )
    # All step sizes default to 1.0.
    if dt is None:
        dt = 1.0
    if dz is None:
        dz = 1.0
    if dy is None:
        dy = dx or 1.0
    if dx is None:
        dx = dy or 1.0
    # Load the file as a dask array.  Treat the uint12 case specially.
    shape = (T, C, Z, Y, X)
    chunks = ("auto", "auto", None, None, None)
    if dtype == "uint12":
        nbytes = (nexpected * 12) // 8
        mmap_array = np.memmap(path, dtype=np.uint8, mode="r", shape=(nbytes,))
        dask_array = da.from_array(mmap_array)
        a = dask_array[0::3].astype(np.uint16)
        b = dask_array[1::3].astype(np.uint16)
        c = dask_array[2::3].astype(np.uint16)
        assert len(a) == len(b) == len(c)
        evens = ((b & 0x0F) << 8) | a
        odds = ((b & 0xF0) >> 4) | (c << 4)
        flat = da.stack([evens, odds], axis=1).ravel()
        dask_array = flat.reshape((T, C, Z, Y, X), chunks=chunks)
    else:
        dtype = np.dtype(dtype)
        mmap_array = np.memmap(path, dtype=dtype, mode="r", shape=shape)
        dask_array = da.from_array(mmap_array, chunks=chunks)  # type: ignore
    # Wrap the dask array as a xarray.DataArray and return it.
    return xr.DataArray(
        dask_array,
        coords={
            "T": dt * np.arange(T),
            "C": range(C),
            "Z": dz * np.arange(Z),
            "Y": dy * np.arange(Y),
            "X": dx * np.arange(X),
        },
    )


def load_raw_chunk(
    path: os.PathLike,
    shape: tuple[int, ...],
    dtype: npt.DTypeLike,
    offset: int,
) -> np.ndarray:
    """
    Load a portion of the supplied raw file as a Numpy array.

    Parameters
    ----------
    path : os.PathLike
        Path (local or remote) to the raw file.
    shape : tuple[int, ...]
        Desired shape of the returned array (e.g. ``(T, C, Z, Y, X)``).  The
        product of the dimensions must match the number of items that will be
        read.
    dtype : npt.DTypeLike
        Numpy data type of the stored items.
    offset : int
        Index of the first element to read **in items**, *not* in bytes.

    Returns
    -------
    np.ndarray
        A dense NumPy array with the requested shape and dtype.
    """
    count = math.prod(shape)
    with fsspec.open(path, newline="") as f:
        return np.fromfile(
            f,  # type: ignore
            dtype=dtype,
            offset=offset,
            count=count,
        ).reshape(shape)
