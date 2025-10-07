import numpy as np
import xarray as xr

from toolsandogh import canonicalize_video, differential_rolling_average, rolling_sum


def test_rolling_sum() -> None:
    r1 = [[1, 2, 3, 4, 5, 6], [9, 8, 7, 6, 5, 4]]
    r2 = [[3, 5, 7, 9, 11], [17, 15, 13, 11, 9]]
    r3 = [[6, 9, 12, 15], [24, 21, 18, 15]]
    r4 = [[10, 14, 18], [30, 26, 22]]
    r5 = [[15, 20], [35, 30]]
    r6 = [[21], [39]]

    # Test rolling sum of various window sizes.
    video = canonicalize_video(np.array(r1, dtype=np.uint8))
    for window_size, expected in enumerate([r1, r2, r3, r4, r5, r6], start=1):
        rsum = rolling_sum(video, window_size, dim="X")
        assert xr.DataArray.equals(rsum, canonicalize_video(expected))

    # Test rolling sum of various data types.
    d1, D1 = np.uint16, np.uint32
    d2, D2 = np.int16, np.int32
    d3, D3 = np.float32, np.float32
    d4, D4 = np.float64, np.float64
    d5, D5 = np.complex64, np.complex64
    d6, D6 = np.complex128, np.complex128

    for d, D in zip([d1, d2, d3, d4, d5, d6], [D1, D2, D3, D4, D5, D6]):
        v = video.astype(d)
        # Rolling sum of size 1 should preserve the dtype.
        rsum = rolling_sum(v, 1, dim="X")
        assert rsum.dtype == d

        # Rolling sum of size >1 should upgrade integer dtypes.
        rsum = rolling_sum(v, 2, dim="X")
        assert rsum.dtype == D


def test_dra() -> None:
    video = canonicalize_video([[1, 2, 3, 4, 5, 6], [9, 8, 7, 6, 5, 4]])
    assert video.shape == (1, 1, 1, 2, 6)
    for window_size, expected in enumerate(
        [[[1, 1, 1, 1, 1], [-1, -1, -1, -1, -1]], [[2, 2, 2], [-2, -2, -2]], [[3], [-3]]], start=1
    ):
        dra = differential_rolling_average(video, window_size, dim="X")
        assert xr.DataArray.equals(dra, canonicalize_video(expected))
