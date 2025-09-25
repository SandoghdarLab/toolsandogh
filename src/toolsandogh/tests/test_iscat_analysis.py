import argparse
import contextlib
import io
import itertools
import math
import pathlib
import shlex
from multiprocessing import Pool

import numpy as np
import xarray as xr

import toolsandogh.scripts.iscat_analysis as iscat_analysis


def test_SharedArray():
    dtypes = [np.float32, np.complex64, np.int32]
    shapes: list[tuple[int, ...]] = [(2,), (2, 3), (2, 3, 5)]
    for dtype in dtypes:
        for shape in shapes:
            # Test individual array access
            sa = iscat_analysis.SharedArray(shape, dtype=dtype)
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

                pool.starmap(initialize, [(sa, i, dtype(n)) for n, i in enumerate(indices)])
            for n, index in enumerate(indices):
                value = dtype(n)
                assert sa[*index] == value


def test_Analysis():
    """Test Analysis class creation and finalization."""
    # Create some test video data
    parent = pathlib.Path(__file__).resolve().parent
    testfile = str(parent / "testfile.tiff")
    a1 = argparse.Namespace(
        input_file=testfile,
        initial_frame=0,
        frames=1,
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
        rvt_limit=1.0,
        loc_radius=1,
        loc_min_mass=0.0,
        loc_percentile=75,
        circle_alpha=1.0,
    )
    a2 = argparse.Namespace(
        input_file=testfile,
        initial_frame=0,
        frames=1,
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
        rvt_limit=1.0,
        loc_radius=2,
        loc_min_mass=0.1,
        loc_percentile=75,
        circle_alpha=0.8,
    )

    for args in [a1, a2]:
        video = np.random.random((10, 32, 32)).astype(np.float32)
        xarr = xr.DataArray(video, dims=("T", "Y", "X"))
        with iscat_analysis.Analysis(args, xarr) as analysis:
            # Ensure that arguments are print-read consistent
            with contextlib.redirect_stdout(io.StringIO()) as target:
                analysis.print_args()
            parser = iscat_analysis.argument_parser()
            argv = shlex.split(target.getvalue())
            args = parser.parse_args(argv[1:])
            iscat_analysis.validate_arguments(parser, args)
            for var in vars(analysis.args):
                assert getattr(args, var) == getattr(analysis.args, var)

            # Run the analysis to completion
            analysis.finish()
