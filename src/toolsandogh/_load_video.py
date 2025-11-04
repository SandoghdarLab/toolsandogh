import ast
import os
import os.path
import pathlib
import re
import urllib.parse
import warnings
from typing import Literal

import bioio
import bioio_czi
import bioio_nd2
import dask
import dask.array as da
import fsspec
import numpy as np
import numpy.typing as npt
import xarray as xr
from fsspec.utils import math

from ._canonicalize_video import canonicalize_video


def load_video(
    path: str | os.PathLike,
    scene: str | int | None = None,
    T: int | None = None,
    C: int | None = None,
    Z: int | None = None,
    Y: int | None = None,
    X: int | None = None,
    dt: float | None = None,
    dz: float | None = None,
    dy: float | None = None,
    dx: float | None = None,
    dtype: npt.DTypeLike | Literal["uint12"] | None = None,
    pylablib_settings_file: str | os.PathLike | bool | None = None,
) -> xr.DataArray:
    """
    Load a video from the given path.

    The resulting video is a 5D xarray with dimensions `TCZYX`.  If any keyword
    arguments are supplied, this function asserts that the resulting video
    matches these arguments, e.g., `Z=1` may be used assert that the result is
    a 2D video.

    The suffix of the supplied path determines the method that is used for
    reading the file.  For ``.raw`` and ``.bin`` files, the `X`, `Y` arguments
    are mandatory, the number of frames `T` is inferred automatically, and the
    remaining arguments default to `Z=1`, `C=1`, `dz=1.0`, `dy=dx=1.0`,
    `dt=1.0`, and `dtype=uint8`.

    Parameters
    ----------
    path : str or os.PathLike
        The name of a file, a URI, or a path.
    scene : str or int
        The name or index of a scene to load from the supplied path.  It is an
        error to supply a name or index that doesn't exist.  When a path
        contains multiple scenes and this argument is not supplied, warn and
        select the first scene.
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
        The expected dtype of the video, or the string `"uint12"`.
    pylablib_settings_file : str or os.Pathlike
        A configuration file as described in :url:`https://pylablib-cam-control.readthedocs.io/en/latest/settings_file.html#settings-file-general`.
        This file is then used to infer `T`, `Y`, `X`, and `dtype` arguments
        that have not been supplied explicitly.
        A value of `True` means that the name of the settings file should be
        derived from the name of the `path` argument by stripping the suffix
        and possible trailing counter and appending `_settings.dat`.

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

    # Handle the case where pylablib_settings_file is True.
    if pylablib_settings_file is True:
        pylablib_settings_file = find_pylablib_settings_file(path)

    # Consult the pylablib_settings_file if it exists.
    if pylablib_settings_file:
        for k, v in parse_pylablib_settings_file(pylablib_settings_file).items():
            if not kwargs.get(k):
                kwargs[k] = v

    print(kwargs)

    # Load the path.
    protocols = fsspec.available_protocols()
    match (url.scheme or "file", path.suffix):
        case (scheme, ".bin" | ".raw") if scheme in protocols:
            if scene is not None:
                raise RuntimeError("Cannot select scenes from raw files.")
            return load_raw_video(path, **kwargs)
        case (_, _) if dtype == "uint12":
            raise RuntimeError("Packed 12-bit integers can only be loaded from raw files.")
        case (_, ".czi"):
            image = bioio.BioImage(pathstr, reader=bioio_czi.Reader)
        case (_, ".nd2"):
            image = bioio.BioImage(pathstr, reader=bioio_nd2.Reader)
        case (_, _):
            # Let bioio figure out the rest or raise an exception
            image = bioio.BioImage(pathstr)

    # Select the right scene.
    scenes = image.scenes
    if len(scenes) == 0:
        raise RuntimeError(f"Zero scenes found in {path}.")
    if isinstance(scene, int) or isinstance(scene, str):
        image.set_scene(scene)
    elif scene is None:
        first_scene = image.scenes[0]
        if len(scenes) > 1:
            warnings.warn(f"Multiple scenes found in {path}, selecting {first_scene}.")
        image.set_scene(first_scene)
    else:
        raise TypeError(f"Invalid scene designator {scene}")

    # Extract the video.
    video = image.xarray_dask_data

    # Ensure the video contains only metadata from the current scene.
    pass  # TODO

    # Ensure the video is canonical and return.
    return canonicalize_video(video, **kwargs)


def load_raw_video(
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
    dtype: npt.DTypeLike | Literal["uint12"] | None = None,
) -> xr.DataArray:
    """
    Load a video from the specified raw file.

    Parameters
    ----------
    path : str or os.PathLike
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

    # Determine size of one array item.
    if dtype is None:
        bits_per_item = 8
    elif dtype == "uint12":
        bits_per_item = 12
    else:
        bits_per_item = np.dtype(dtype).itemsize * 8

    # The number of frames T can be inferred from the file's size.
    max_bytes = os.path.getsize(path)
    max_items = (max_bytes * 8) // bits_per_item
    items_per_frame = C * Z * Y * X
    if T is None:
        T = max_items // items_per_frame
        if T == 0:
            raise RuntimeError("Video is empty.")
    nitems = T * items_per_frame
    if nitems > max_items:
        raise RuntimeError(
            f"The file {path} has only {max_items} items, but {nitems} were expected."
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

    # Determine a reasonable Dask chunk size.
    bits_per_chunk = 128 * 1024 * 1024 * 8
    bits_per_frame = items_per_frame * bits_per_item
    frames_per_chunk = max(1, bits_per_chunk // bits_per_frame)

    # Distinguish the packed 12-bit case from all others.
    if bits_per_item == 12:
        # Ensure chunks of packed 12-bit data are byte-aligned.
        if ((frames_per_chunk * bits_per_chunk) % 8) != 0:
            frames_per_chunk += 1
        bits_per_chunk = frames_per_chunk * items_per_frame * bits_per_item
        assert (bits_per_chunk % 8) == 0
        bytes_per_chunk = bits_per_chunk // 8
        nbytes = math.ceil((bits_per_item * nitems) / 8)
        array = load_raw_array(path, shape=(nbytes,), dtype=np.uint8, chunk_size=bytes_per_chunk)
        a = array[0::3].astype(np.uint16)
        b = array[1::3].astype(np.uint16)
        c = array[2::3].astype(np.uint16)
        assert len(a) == len(b) == len(c)
        evens = ((b & 0x0F) << 8) | a
        odds = ((b & 0xF0) >> 4) | (c << 4)
        flat = da.stack([evens, odds], axis=1).ravel()
        dask_array = flat.reshape((T, C, Z, Y, X))
    else:
        shape = (T, C, Z, Y, X)
        dask_array = load_raw_array(path, shape=shape, dtype=dtype, chunk_size=frames_per_chunk)

    # Wrap the dask array as a xarray.DataArray and return it.
    return canonicalize_video(dask_array)  # type: ignore


def load_raw_array(
    path: str | os.PathLike,
    shape: tuple[int, ...],
    dtype: npt.DTypeLike,
    *,
    chunk_size: int = 1,
) -> da.Array:
    """
    Lazily load a binary file as a Dask array.

    The file is interpreted as a flat, C‑ordered array of `dtype` and
    reshaped to `shape`.  The array is partitioned **only** along the
    leading axis; each partition is read on‑demand via :func:`load_raw_chunk`
    and then stacked back together.

    Parameters
    ----------
    path : str
        Path to the binary file.
    shape : tuple[int, ...]
        Desired shape of the resulting array.
    dtype : str or np.dtype
        Data type of the stored values.
    chunk_size : int, optional
        Size of the chunk along axis 0.  Defaults to 1.

    Returns
    -------
    dask.array.Array
        A Dask array whose blocks are produced by :func:`load_raw_chunk` and
        concatenated along axis zero.

    Notes
    -----
    * The function assumes the file is stored in C‑order (row‑major).
    * The byte offset for a slice ``i`` is computed as::

          offset_i = i * (product(shape[1:]) * dtype.itemsize)

      because each slice along axis 0 occupies exactly that many bytes.
    * ``chunks`` may be larger than the length of axis 0; in that case a
      single block covering the whole axis is created.
    """
    # Normalize inputs.
    dtype = np.dtype(dtype)
    if len(shape) == 0:
        raise ValueError("`shape` must contain at least one dimension")
    n = shape[0]

    # Create delayed objects for each block.
    delayed_blocks = []
    block_shapes = []
    slice_bytes = math.prod(shape[1:]) * dtype.itemsize
    for start, length in ((i, min(chunk_size, n - i)) for i in range(0, n, chunk_size)):
        block_shape = (length,) + shape[1:]
        block_shapes.append(block_shape)
        offset = start * slice_bytes

        # Wrap the low-level loader with dask.delayed:
        delayed_block = dask.delayed(load_raw_chunk)(path, block_shape, dtype, offset)  # type: ignore
        delayed_blocks.append(delayed_block)

    # Turn delayed objects into Dask array blocks.
    dask_blocks = [
        da.from_delayed(block, shape=shape, dtype=dtype)
        for block, shape in zip(delayed_blocks, block_shapes)
    ]

    return da.concatenate(dask_blocks, axis=0)


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
        Index of the first byte to read.

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


def find_pylablib_settings_file(path: str | os.PathLike) -> os.PathLike | None:
    path = pathlib.Path(path)
    parent = path.parent
    stem = path.stem

    if (candidate := parent / f"{stem}_settings.dat").exists():
        return candidate

    pattern = r"_\d+$"
    if re.search(pattern, stem):
        new_stem = re.sub(pattern, "", stem)
        print(new_stem)
        if (candidate := parent / f"{new_stem}_settings.dat").exists():
            return candidate

    return None


def parse_pylablib_settings_file(path: str | os.PathLike) -> dict:
    path = pathlib.Path(path)
    if not path.exists():
        warnings.warn(f"Found no pylablib settings file at {path}.")
        return {}

    attrs = {}
    with open(path, "r") as file:
        for line in file:
            line = line.strip()

            # Skip comments
            if line.startswith("#"):
                continue

            parts = line.split(None, 1)

            # Skip empty lines and keys with no values
            if len(parts) < 2:
                continue

            key, value = parts[0], parts[1]

            attrs[key] = value

    kwargs = {}

    if value := attrs.get("cam/data_dimensions"):
        kwargs["Y"], kwargs["X"] = ast.literal_eval(value)

    if last := attrs.get("save/last_frame_index"):
        first = attrs.get("save/first_frame_index", "0")
        total = int(last) + 1 - int(first)
        if value := attrs.get("save/chunk_size"):
            chunk_size = ast.literal_eval(value)
            if isinstance(chunk_size, int):
                kwargs["T"] = chunk_size
        else:
            kwargs["T"] = total

    if dtype := attrs.get("save/frame/dtype"):
        kwargs["dtype"] = np.dtype(dtype)

    return kwargs
