import argparse
import itertools
import math
from multiprocessing import Pool

import numpy as np
import xarray as xr

from toolsandogh.iscat_analysis import Analysis, SharedArray


def test_SharedArray():
    dtypes = [np.float32, np.complex64, np.int32]
    shapes: list[tuple[int, ...]] = [(2,), (2, 3), (2, 3, 5)]
    for dtype in dtypes:
        for shape in shapes:
            # Test individual array access
            sa = SharedArray(shape, dtype=dtype)
            indices = itertools.product(*[list(range(dim)) for dim in shape])
            for index in indices:
                value = dtype(np.random.randint(256))
                sa[*index] = value
                assert sa[*index] == value

            # Fill array with zeros
            sa[...] = dtype(0)

            # Test concurrent access
            size = math.prod(shape)
            with Pool(processes=size) as pool:

                def initialize(array, index, value):
                    array[*index] = value

                pool.starmap(
                    initialize, [(sa, i, dtype(n)) for n, i in enumerate(indices)]
                )
            for n, index in enumerate(indices):
                value = dtype(n)
                assert sa[*index] == value


def test_Analysis():
    """Test Analysis class creation and finalization."""
    # Create some test video data
    a1 = argparse.Namespace(
        processes=1,
        rvt_upsample=1,
        particles=2,
        fft_inner_radius=0.0,
        fft_outer_radius=1.0,
        fft_row_noise_threshold=0.00,
        fft_column_noise_threshold=0.00,
        dra_window_size=0,
        rvt_min_radius=1,
        rvt_max_radius=2,
        tracking_radius=1,
        tracking_min_mass=0.0,
        tracking_percentile=75,
        circle_alpha=1.0,
    )
    a2 = argparse.Namespace(
        processes=2,
        rvt_upsample=2,
        particles=42,
        fft_inner_radius=0.1,
        fft_outer_radius=0.9,
        fft_row_noise_threshold=0.01,
        fft_column_noise_threshold=0.01,
        dra_window_size=5,
        rvt_min_radius=1,
        rvt_max_radius=3,
        tracking_radius=2,
        tracking_min_mass=0.1,
        tracking_percentile=75,
        circle_alpha=0.8,
    )

    for args in [a1, a2]:
        video = np.random.random((10, 32, 32)).astype(np.float32)
        xarr = xr.DataArray(video, dims=("T", "Y", "X"))
        with Analysis(args, xarr) as analysis:
            analysis.finish()
