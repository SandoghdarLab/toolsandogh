import os
import pathlib
import urllib.parse

import bioio_imageio.writers
import xarray as xr

from ._validate_video import validate_video


def store_video(video: xr.DataArray, path: os.PathLike) -> None:
    """
    Store a given microscopy dataset at the supplied path.

    Parameters
    ----------
    video : xarray.DataArray
        A canonical TCZYX array.
    path : os.PathLike
        The name of a file, a URI, or a path.

    Returns
    -------
    os.PathLike
        The name of a file, a URI, or a path.
    """
    validate_video(video)
    # Decode the path
    pathstr = str(path)
    url = urllib.parse.urlparse(pathstr)
    path = pathlib.Path(url.path)

    match (url.scheme or "file", path.suffix):
        case (scheme, ".bin" | ".raw"):
            pass  # TODO
        case (scheme, ".mp4" | ".avi"):
            data = video.stack(F=("T", "C", "Z")).transpose("F", "Y", "X").data
            bioio_imageio.writers.TimeseriesWriter.save(data, pathstr, dimorder="TYX")

    scheme = url.scheme or "file"
    suffix = path.suffix
    if scheme == "file":
        if suffix in [".bin", ".raw"]:
            pass
        elif suffix == ".czi":
            pass
        elif suffix == ".nd2":
            pass
    else:
        raise RuntimeError(f"Don't know how to store data via {scheme}.")
