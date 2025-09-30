import numpy as np
import numpy.typing as npt
import xarray as xr

from ._validate_video import validate_video


def canonicalize_video(video: npt.ArrayLike):
    """
    Turn the supplied data into its canonical TCZYX video representation.

    Parameters
    ----------
    video : xr.DataArray
        An xarray.

    Returns
    -------
    xarray.DataArray
        A TCZYX video with the supplied parameters.
    """
    # Turn any non-xarray into an xarray.
    if not isinstance(video, xr.DataArray):
        video = xr.DataArray(video)

        # Determine the appropriate dims
        rank = len(video.shape)
        match rank:
            case 0:
                dims = ()
            case 1:
                dims = ("X",)
            case 2:
                dims = ("Y", "X")
            case 3:
                dims = ("T", "Y", "X")
            case 4:
                dims = ("T", "Z", "Y", "X")
            case 5:
                dims = ("T", "C", "Z", "Y", "X")
            case 6:
                dims = ("T", "C", "Z", "Y", "X", "S")
            case _:
                raise RuntimeError(f"Cannot interpret {rank}-dimensional data as a video.")
        video = xr.DataArray(video.data, dims=dims)

    # Ensure the TZYX axes exist and are continuous.
    for dim in ("T", "Z", "Y", "X"):
        if dim not in video.dims:
            video = video.expand_dims({dim: np.arange(1.0)})
        elif video[dim].dtype != np.float64:
            values = np.array(video[dim])
            video = video.assign_coords({dim: values.astype(np.float64)})

    # If there is a S axis, merge its entries.
    if "S" in video.dims:
        video = video.mean("S", dtype=np.float32).astype(video.dtype)

    # Ensure the correct ordering of axes.
    video = video.transpose("T", "C", "Z", "Y", "X")

    # Raise an exception if the video is still not in canonical form.
    validate_video(video)

    # Done.
    return video
