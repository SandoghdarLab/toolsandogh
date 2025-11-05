import os
import pathlib
import urllib.parse

import numpy.typing as npt
import xarray as xr
from bioio_imageio.writers import TimeseriesWriter
from bioio_ome_tiff.writers import OmeTiffWriter
from bioio_ome_zarr.writers import OMEZarrWriter

from ._canonicalize_video import canonicalize_video


def store_video(video: npt.ArrayLike, path: str | os.PathLike) -> None:
    """
    Store a given microscopy dataset at the supplied path.

    Parameters
    ----------
    video : npt.ArrayLike
        An array that is a suitable argument to :py:func:`~toolsandogh.canonicalize_video`.
    path : os.PathLike
        The name of a file, a URI, or a path.
    """
    video = canonicalize_video(video)

    # Decode the path
    pathstr = str(path)
    url = urllib.parse.urlparse(pathstr)
    path = pathlib.Path(url.path)

    match (url.scheme or "file", path.suffix):
        case (scheme, ".bin" | ".raw"):
            store_raw_video(video, path)
        case (scheme, ".mp4" | ".avi"):
            data = video.stack(F=("T", "C", "Z")).transpose("F", "Y", "X").data
            TimeseriesWriter.save(data, pathstr, dimorder="TYX")
        case (scheme, ".tiff"):
            OmeTiffWriter.save(video.to_numpy(), pathstr)
        case (scheme, ".zarr"):
            writer = OMEZarrWriter(
                store=pathstr,
                level_shapes=video.shape,
                dtype=video.dtype,
                zarr_format=3,
            )
            writer.write_full_volume(video.data)
        case ("file", suffix):
            raise RuntimeError(f"Don't know how to store {suffix} data.")
        case (scheme, suffix):
            raise RuntimeError(f"Don't know how to store {suffix} data via {scheme}.")


def store_raw_video(video: xr.DataArray, path: str | os.PathLike) -> None:
    """
    Store a given microscopy dataset as a binary file.

    This method of storing discards all metadata, including the shape, so it is
    usually better to chose a different representation, e.g., as a .zarr file.

    Parameters
    ----------
    video : xarray.DataArray
        A canonical TCZYX array.
    path : os.PathLike
        The name of a file, a URI, or a path.
    """
    # TODO write chunk after chunk instead of allocating the entire video as
    # one contiguous Numpy array.
    video.to_numpy().tofile(path)
