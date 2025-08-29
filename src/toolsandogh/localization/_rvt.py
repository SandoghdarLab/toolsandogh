from typing import Literal

import imgrvt
import numpy as np
import xarray as xr

_rvt_pad_modes = Literal[
    "constant",
    "edge",
    "reflect",
    "linear_ramp",
    "maximum",
    "mean",
    "median",
    "minimum",
    "reflect",
    "symmetric",
    "wrap",
    "empty",
]


def radial_variance_transform(
    data: xr.DataArray,
    rmin: int,
    rmax: int,
    upsample: int = 1,
    rweights: np.ndarray | None = None,
    coarse_factor: int = 1,
    coarse_mode: Literal["add", "skip"] = "add",
    pad_mode: _rvt_pad_modes = "constant",
    kind: Literal["basic", "normalized"] = "normalized",
) -> xr.DataArray:
    """
    Perform the Radial Variance Transform across the XY-dimension.

    Args:
        data (xarray.DataArray): A TCZYX DataArray.
        rmin (int): Minimal radius (inclusive).
        rmax (int): Maximal radius (inclusive).
        kind (str, optional): Either `"basic"` (Variance of Mean) or `"normalized"` (Variance of Mean / Mean of Variance).
            The normalized version increases subpixel bias but works better at lower SNR. Defaults to "normalized".
        highpass_size (int, optional): Size of the high-pass filter. `None` means no filter (effectively, infinite size). Defaults to None.
        upsample (int, optional): Integer image upsampling factor. `rmin` and `rmax` are adjusted automatically
            (i.e., they refer to the non-upsampled image). If `upsample > 1`, the resulting X and Y sizes are multiplied by `upsample`.
            Defaults to 1.
        rweights (numpy.ndarray, optional): Relative weights of different radial kernels. Must be a 1D array of length
            `(rmax - rmin + 1) // coarse_factor`. Defaults to None.
        coarse_factor (int, optional): The reduction factor for the number of ring kernels. Can be used to speed up calculations
            at the expense of precision. Defaults to 1.
        coarse_mode (str, optional): The reduction method. Can be `"add"` (add `coarse_factor` rings in a row to get a thicker ring,
            which works better for smooth features) or `"skip"` (only keep one in `coarse_factor` rings, which works better for
            very fine features). Defaults to "add".
        pad_mode (str, optional): Edge padding mode for convolutions. Can be either one of the modes accepted by `np.pad`
            (such as `"constant"`, `"reflect"`, or `"edge"`), or `"fast"`, which means faster no-padding (a combination of
            `"wrap"` and `"constant"` padding depending on the image size). `"fast"` mode works faster for smaller images and
            larger `rmax`, but the border pixels (within `rmax` from the edge) are less reliable. Note that the image mean is
            subtracted before processing, so `pad_mode="constant"` (default) is equivalent to padding with a constant value
            equal to the image mean. Defaults to "constant".

    Returns:
        xarray.DataArray: A TCZYX DataArray whose X and Y axis are multiplied by `upsample`.
    """
    return xr.apply_ufunc(
        imgrvt.rvt,
        data,
        input_core_dims=[["Y", "X"]],
        output_core_dims=[["Y", "X"]],
        output_dtypes=[np.float64],
        dask="parallelized",
        vectorize=True,
        dask_gufunc_kwargs={
            "allow_rechunk": True,
            "output_sizes": {
                "Y": data.sizes["Y"] * upsample,
                "X": data.sizes["X"] * upsample,
            },
        },
        kwargs={
            "rmin": rmin,
            "rmax": rmax,
            "upsample": upsample,
            "rweights": rweights,
            "coarse_factor": coarse_factor,
            "coarse_mode": coarse_mode,
            "pad_mode": pad_mode,
            "kind": kind,
        },
    )
