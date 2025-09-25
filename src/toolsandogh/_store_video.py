import os
import pathlib
import urllib.parse

import xarray as xr

from ._validate_video import validate_video


def store_video(dataset: xr.DataArray, path: os.PathLike) -> None:
    """
    Store a given microscopy dataset at the supplied path.

    Parameters
    ----------
    dataset : xarray.DataArray
        A canonical TCZYX array.
    path : os.PathLike
        The name of a file, a URI, or a path.

    Returns
    -------
    os.PathLike
        The name of a file, a URI, or a path.
    """
    validate_video(dataset)
    # Decode the path
    pathstr = str(path)
    url = urllib.parse.urlparse(pathstr)
    path = pathlib.Path(url.path)
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
