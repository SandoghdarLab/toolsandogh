from typing import Literal

import dask.array as da
import numpy as np
import xarray as xr

from ._canonicalize_video import canonicalize_video

Dim = Literal["T", "C", "Z", "Y", "X"]

INTEGER_DTYPES: list[type] = [
    np.uint8,
    np.int8,
    np.uint16,
    np.int16,
    np.uint32,
    np.int32,
    np.uint64,
    np.int64,
]


def rolling_sum(
    video: xr.DataArray,
    window_size: int,
    dim: Dim = "T",
) -> xr.DataArray:
    """
    Return the rolling sum of a video along an axis.

    Parameters
    ----------
    video : xarray.DataArray,
        The video to be summed.
    window_size : int
        The size of the moving window.
    dim : str
        The name of the axis being summed along.

    Returns
    -------
    xarray.DataArray
        A new video whose contents are the result of the rolling sum operation.
    """
    video = canonicalize_video(video)

    # Check the window_size parameter.
    if window_size <= 0:
        raise ValueError(f"Window size must be positive, got {window_size}.")
    n = len(video[dim])
    k = n - window_size + 1
    if k <= 0:
        raise ValueError(f"Window size {window_size} too large for axis of length {n}.")

    # Choose a suitable dtype for the summation.
    if np.issubdtype(video.dtype, np.integer):
        vinfo = np.iinfo(video.dtype)
        vmin = window_size * vinfo.min
        vmax = window_size * vinfo.max
        dtype = None
        for integer_dtype in INTEGER_DTYPES:
            dinfo = np.iinfo(integer_dtype)
            if (dinfo.min <= vmin) and (vmax <= dinfo.max):
                dtype = integer_dtype
                break
        if dtype is None:
            dtype = np.float64
    else:
        dtype = video.dtype

    # Pad with zeros.
    axis = video.get_axis_num(dim)
    axes = range(video.ndim)
    pad_width = [(1, 0) if i == axis else (0, 0) for i in axes]

    # Cut the data into overlapping chunks.
    parts = da.map_overlap(
        lambda a: a,
        da.pad(video.data, pad_width, mode="constant", constant_values=0),
        depth={axis: window_size - 1},
        boundary="none",
        dtype=dtype,
    )
    parts.persist()
    cumsum = da.cumsum(parts, axis=axis, method="blelloch", dtype=dtype)
    right = [slice(window_size, None) if i == axis else slice(None) for i in axes]
    left = [slice(None, -window_size) if i == axis else slice(None) for i in axes]
    return canonicalize_video(cumsum[*right] - cumsum[*left])


def rolling_average(
    video: xr.DataArray,
    window_size: int,
    dim: Dim = "T",
) -> xr.DataArray:
    """
    Return the rolling average along an axis.

    Parameters
    ----------
    video : xarray.DataArray,
        The video for which to compute the rolling average.
    window_size : int
        The size of the moving window.
    dim : str
        The name of the axis being averaged over.

    Returns
    -------
    xarray.DataArray
        A new video whose contents are the result of the rolling average operation.
    """
    rsum = rolling_sum(video, window_size=window_size, dim=dim)

    # Determine the element type.
    if np.issubdtype(video.dtype, np.integer):
        dtype = np.float32
    else:
        dtype = video.dtype

    return rsum.astype(dtype) / window_size


def differential_rolling_average(
    video: xr.DataArray,
    window_size: int,
    dim: Dim = "T",
) -> xr.DataArray:
    """
    Return the differential rolling average along an axis.

    Parameters
    ----------
    video : xarray.DataArray,
        The video for which to compute the differential rolling average.
    window_size : int
        The batch size of each operand of the differential.
    dim : str
        The name of the axis being operated on.

    Returns
    -------
    xarray.DataArray
        A new video whose contents are the result of the differential rolling average operation.
    """
    video = canonicalize_video(video)

    # Check the window_size parameter.
    if window_size <= 0:
        raise ValueError(f"Window size must be positive, got {window_size}.")

    ra = rolling_average(video, window_size=window_size, dim=dim)
    left = ra.isel({dim: slice(None, -window_size)})
    right = ra.isel({dim: slice(window_size, None)})
    return canonicalize_video(right.data - left.data)
