import os

import xarray as xr

from ._verify import verify_video


def store_video(dataset: xr.DataArray, path: os.PathLike) -> None:
    """
    Store a given microscopy dataset at the supplied path.

    Args:
        dataset (xarray.DataArray): A TCZYX DataArray
        path (os.PathLike): The name of a file, a URI, or a path.

    Returns:
        os.PathLike: The name of a file, a URI, or a path.
    """
    verify_video(dataset)
    pass  # TODO
