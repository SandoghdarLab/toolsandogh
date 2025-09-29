import numpy as np
import xarray as xr


def canonicalize_video(video: xr.DataArray):
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

    # Transpose and return.
    return video.transpose("T", "C", "Z", "Y", "X")
