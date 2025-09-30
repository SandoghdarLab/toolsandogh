"""Tests input and output of videos."""

import pathlib
import tempfile

import imageio.v3 as iio
import numpy as np

from toolsandogh import canonicalize_video, generate_video, load_video, store_video


def test_generate_video() -> None:
    """Unit test for :func:`toolsandogh.generate_video`."""
    # Create a video with default parameters.
    v1 = generate_video().load()
    assert len(v1.to_numpy().shape) == 5

    # Create a video with custom parameters.
    v2 = generate_video(T=5, Z=3, Y=2, X=1).load()
    assert len(v2["T"]) == 5
    assert len(v2.Z) == 3
    assert len(v2.Y) == 2
    assert len(v2.X) == 1


def test_tiff_io() -> None:
    """Test conversion of videos to/from .tiff files."""
    # Create arrays of varying dtypes.
    shape = (2, 3, 5, 7, 11)
    rng = np.random.default_rng()
    arrays = [
        rng.integers(0, 2**8, size=shape, dtype=np.uint8),
        rng.integers(-(2**15), (2**15), size=shape, dtype=np.int16),
        rng.random(size=shape, dtype=np.float32),
    ]
    videos = [canonicalize_video(array) for array in arrays]

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        for n, video in enumerate(videos):
            path = pathlib.Path(tmpdir) / f"array{n}.tiff"
            store_video(video, path)
            other_video = load_video(path)
            # With .tiff, the resulting video should be bitwise identical.
            assert (video == other_video).all()


def test_zarr_io() -> None:
    """Test conversion of videos to/from .zarr files."""
    # Create arrays of varying dtypes.
    shape = (2, 3, 5, 7, 11)
    rng = np.random.default_rng()
    arrays = [
        rng.integers(0, 2**8, size=shape, dtype=np.uint8),
        rng.integers(-(2**15), (2**15), size=shape, dtype=np.int16),
        rng.random(size=shape, dtype=np.float32),
    ]
    videos = [canonicalize_video(array) for array in arrays]

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        for n, video in enumerate(videos):
            path = pathlib.Path(tmpdir) / f"array{n}.zarr"
            store_video(video, path)
            other_video = load_video(path)
            # With .zarr, the resulting video should be bitwise identical.
            assert (video == other_video).all()


def test_mp4_io() -> None:
    """Test conversion of videos to/from .mp4 files."""
    # Create test data
    shape = (64, 32, 16, 3)
    data = np.random.randint(0, 256, size=shape, dtype=np.uint8)
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        for suffix in [".mp4", ".avi"]:
            # Write data to disc.
            path = pathlib.Path(tmpdir) / f"imwrite{suffix}"
            iio.imwrite(path, data, fps=24, pixelformat="gray")

            # Ensure that load_video behaves the same as iio.imread.
            expected = iio.imread(path)
            video = load_video(path)
            video_data = video.isel(Z=0).transpose("T", "Y", "X", "C").to_numpy()
            assert np.all(video_data == expected)

            # Ensure that store_video behaves the same as iio.imwrite.
            vidpath = pathlib.Path(tmpdir) / f"store_video{suffix}"
            store_video(video, vidpath)
            video_data = iio.imread(vidpath)
            assert np.mean(np.abs(np.float32(video_data) - np.float32(expected))) < 10.0


def test_load_video() -> None:
    """Unit test for :func:`toolsandogh.load_video`."""
    parent = pathlib.Path(__file__).resolve().parent
    path = parent / "testfile.tiff"
    load_video(path)
    load_video(str(path))
    load_video("file://" + str(path))


def test_store_video() -> None:
    """Unit test for :func:`toolsandogh.load_video`."""
    pass
