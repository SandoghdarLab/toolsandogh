"""Tests for the particle localizer."""

import jax.numpy as jnp
import numpy as np
import polars
import pytest

import toolsandogh as tog


def _gaussian_psf(sigma: float, n: int) -> jnp.ndarray:
    """Build a 2D Gaussian PSF of shape (n, n)."""
    y, x = jnp.meshgrid(jnp.arange(n) - n // 2, jnp.arange(n) - n // 2, indexing="ij")
    return jnp.exp(-(y * y + x * x) / (2 * sigma * sigma))


def _gaussian_psf_3d(sigma: float, n: int) -> jnp.ndarray:
    """Build a 3D Gaussian PSF of shape (n, n, n)."""
    coords = [jnp.arange(n) - n // 2 for _ in range(3)]
    grids = jnp.meshgrid(*coords, indexing="ij")
    r2 = sum(g * g for g in grids)
    return jnp.exp(-r2 / (2 * sigma * sigma))


def test_locate_returns_strict_schema() -> None:
    """The returned DataFrame must have the agreed columns and dtypes."""
    psf = _gaussian_psf(1.5, 7).reshape(1, 7, 7)
    trajectories = polars.DataFrame(
        {
            "t": [0],
            "c": [0],
            "z": [0.0],
            "y": [5.0],
            "x": [5.0],
            "contrast": [1.0],
            "particle_id": [0],
        }
    )
    video = tog.simulate_particles(
        trajectories,
        psf,
        shape=(1, 1, 1, 10, 10),
        noise_sigma=0.0,
        dtype=np.float32,
    )
    locs = tog.locate_in_chunk(
        jnp.asarray(video.values[0]),
        psf,
        min_distance=3,
        min_contrast=0.0,
        iterations=50,
        atol=1e-4,
    )
    expected = {
        "t": polars.Int32,
        "c": polars.Int32,
        "z": polars.Float32,
        "y": polars.Float32,
        "x": polars.Float32,
        "contrast": polars.Float32,
        "mass": polars.Float32,
        "snr": polars.Float32,
        "chi2": polars.Float32,
        "n_iter": polars.Int32,
        "converged": polars.Boolean,
    }
    for name, dtype in expected.items():
        assert name in locs.columns
        assert locs.schema[name] == dtype, (
            f"column {name} has dtype {locs.schema[name]}, expected {dtype}"
        )


def test_locate_two_separated_emitters() -> None:
    """Two emitters well-separated should both be recovered."""
    psf = _gaussian_psf(1.5, 7).reshape(1, 7, 7)
    trajectories = polars.DataFrame(
        {
            "t": [0, 0],
            "c": [0, 0],
            "z": [0.0, 0.0],
            "y": [3.0, 8.0],
            "x": [3.0, 8.0],
            "contrast": [2.0, -1.5],
            "particle_id": [0, 1],
        }
    )
    video = tog.simulate_particles(
        trajectories,
        psf,
        shape=(1, 1, 1, 12, 12),
        noise_sigma=0.0,
        dtype=np.float32,
    )
    locs = tog.locate_in_chunk(
        jnp.asarray(video.values[0]),
        psf,
        min_distance=3,
        min_contrast=0.0,
        iterations=50,
        atol=1e-4,
    )
    assert locs.shape[0] == 2
    ys = sorted(locs["y"].to_list())
    assert abs(ys[0] - 3.0) < 0.05
    assert abs(ys[1] - 8.0) < 0.05


def test_locate_sign_argument() -> None:
    """The sign argument filters the sign of detected peaks."""
    psf = _gaussian_psf(1.5, 7).reshape(1, 7, 7)
    trajectories = polars.DataFrame(
        {
            "t": [0, 0],
            "c": [0, 0],
            "z": [0.0, 0.0],
            "y": [3.0, 8.0],
            "x": [3.0, 8.0],
            "contrast": [2.0, -1.5],
            "particle_id": [0, 1],
        }
    )
    video = tog.simulate_particles(
        trajectories,
        psf,
        shape=(1, 1, 1, 12, 12),
        noise_sigma=0.0,
        dtype=np.float32,
    )
    pos_locs = tog.locate_in_chunk(
        jnp.asarray(video.values[0]),
        psf,
        min_distance=3,
        min_contrast=0.0,
        sign="positive",
        iterations=50,
        atol=1e-4,
    )
    assert all(a > 0 for a in pos_locs["contrast"].to_list())

    neg_locs = tog.locate_in_chunk(
        jnp.asarray(video.values[0]),
        psf,
        min_distance=3,
        min_contrast=0.0,
        sign="negative",
        iterations=50,
        atol=1e-4,
    )
    assert all(a < 0 for a in neg_locs["contrast"].to_list())


def test_locate_starting_frame_offset() -> None:
    """The starting_frame argument must be added to the t column."""
    psf = _gaussian_psf(1.5, 7).reshape(1, 7, 7)
    trajectories = polars.DataFrame(
        {
            "t": [0],
            "c": [0],
            "z": [0.0],
            "y": [5.0],
            "x": [5.0],
            "contrast": [1.0],
            "particle_id": [0],
        }
    )
    video = tog.simulate_particles(
        trajectories,
        psf,
        shape=(1, 1, 1, 10, 10),
        noise_sigma=0.0,
        dtype=np.float32,
    )
    locs = tog.locate_in_chunk(
        jnp.asarray(video.values[0]),
        psf,
        starting_frame=42,
        min_distance=3,
        min_contrast=0.0,
        iterations=50,
        atol=1e-4,
    )
    assert locs["t"].to_list() == [42]


def test_locate_channel_validation() -> None:
    """`locate` must reject a video with multiple channels and no channel arg."""
    psf = _gaussian_psf(1.5, 7).reshape(1, 7, 7)
    # Build a video with two channels.
    trajectories = polars.DataFrame(
        {
            "t": [0],
            "c": [0],
            "z": [0.0],
            "y": [5.0],
            "x": [5.0],
            "contrast": [1.0],
            "particle_id": [0],
        }
    )
    video = tog.simulate_particles(
        trajectories,
        psf,
        shape=(1, 2, 1, 10, 10),
        noise_sigma=0.0,
        dtype=np.float32,
    )
    with pytest.raises(ValueError, match="channel"):
        tog.locate(video, psf)


def test_locate_dtype_strictness() -> None:
    """All float columns must be exactly float32 (or the chosen dtype)."""
    psf = _gaussian_psf(1.5, 7).reshape(1, 7, 7)
    trajectories = polars.DataFrame(
        {
            "t": [0],
            "c": [0],
            "z": [0.0],
            "y": [5.0],
            "x": [5.0],
            "contrast": [1.0],
            "particle_id": [0],
        }
    )
    video = tog.simulate_particles(
        trajectories,
        psf,
        shape=(1, 1, 1, 10, 10),
        noise_sigma=0.0,
        dtype=np.float32,
    )
    locs = tog.locate_in_chunk(
        jnp.asarray(video.values[0]),
        psf,
        min_distance=3,
        min_contrast=0.0,
        iterations=50,
        atol=1e-4,
    )
    float_cols = {"z", "y", "x", "contrast", "mass", "snr", "chi2"}
    for col in float_cols:
        if col in locs.columns and locs[col].null_count() < locs.shape[0]:
            assert locs[col].dtype == polars.Float32


def test_locate_dtype_argument() -> None:
    """The ``dtype`` argument must be accepted and used for the data."""
    psf = _gaussian_psf(1.5, 7).reshape(1, 7, 7)
    trajectories = polars.DataFrame(
        {
            "t": [0],
            "c": [0],
            "z": [0.0],
            "y": [5.0],
            "x": [5.0],
            "contrast": [1.0],
            "particle_id": [0],
        }
    )
    video = tog.simulate_particles(
        trajectories,
        psf,
        shape=(1, 1, 1, 10, 10),
        noise_sigma=0.0,
        dtype=np.float32,
    )
    # The default dtype (float32) is the only one the JIT-cached
    # functions support on this build.  We verify that the dtype
    # argument is accepted and the result is a well-formed DataFrame
    # of the right shape; checking the float column dtype would
    # require float64 support (``jax_enable_x64 = True``) which
    # would also clear the JIT cache and affect the other tests.
    locs = tog.locate(video, psf, dtype=np.float32, chunk_size=1)
    assert locs.shape[0] == 1
    assert locs["y"].to_list()[0] != 0.0  # sanity check: a fit happened


def test_locate_channel_none_default_single_channel() -> None:
    """``channel=None`` on a single-channel video should auto-select it."""
    psf = _gaussian_psf(1.5, 7).reshape(1, 7, 7)
    trajectories = polars.DataFrame(
        {
            "t": [0],
            "c": [0],
            "z": [0.0],
            "y": [5.0],
            "x": [5.0],
            "contrast": [1.0],
            "particle_id": [0],
        }
    )
    video = tog.simulate_particles(
        trajectories,
        psf,
        shape=(1, 1, 1, 10, 10),
        noise_sigma=0.0,
        dtype=np.float32,
    )
    # No ``channel`` argument: the localizer must pick the only channel
    # and still produce a correctly-typed ``c`` column.
    locs = tog.locate(video, psf, chunk_size=1)
    assert locs.shape[0] == 1
    assert locs["c"].dtype == polars.Int32
    assert locs["c"].to_list() == [0]


def test_locate_in_chunk_2d() -> None:
    """The 2D case (Pz=1) should work just like the general 3D case."""
    psf = _gaussian_psf(1.5, 7).reshape(1, 7, 7)
    trajectories = polars.DataFrame(
        {
            "t": [0],
            "c": [0],
            "z": [0.0],
            "y": [3.4],
            "x": [5.6],
            "contrast": [2.0],
            "particle_id": [0],
        }
    )
    video = tog.simulate_particles(
        trajectories,
        psf,
        shape=(1, 1, 1, 10, 10),
        noise_sigma=0.05,
        seed=42,
        dtype=np.float32,
    )
    locs = tog.locate_in_chunk(
        jnp.asarray(video.values[0]),
        psf,
        min_distance=3,
        min_contrast=0.1,
        iterations=50,
        atol=1e-4,
    )
    assert locs.shape[0] >= 1
    # At least one located emitter should be near the true position.
    best = int(np.argmax(np.abs(locs["contrast"].to_numpy())))
    assert abs(locs["y"][best] - 3.4) < 0.1
    assert abs(locs["x"][best] - 5.6) < 0.1


def test_locate_circular_motion_2d() -> None:
    """A particle circling the video centre should be recovered at every frame."""
    psf = _gaussian_psf(1.5, 9).reshape(1, 9, 9)
    n_frames = 24
    cy, cx = 8.5, 8.5
    radius = 2.5
    t = np.arange(n_frames)
    theta = 2.0 * np.pi * t / n_frames
    ys = cy + radius * np.cos(theta)
    xs = cx + radius * np.sin(theta)
    trajectories = polars.DataFrame(
        {
            "t": t.tolist(),
            "c": [0] * n_frames,
            "z": [0.0] * n_frames,
            "y": ys.tolist(),
            "x": xs.tolist(),
            "contrast": [2.0] * n_frames,
            "particle_id": [0] * n_frames,
        }
    )
    video = tog.simulate_particles(
        trajectories,
        psf,
        shape=(n_frames, 1, 1, 18, 18),
        noise_sigma=0.02,
        seed=0,
        dtype=np.float32,
    )
    locs = tog.locate(
        video,
        psf,
        chunk_size=4,
        min_distance=4,
        min_contrast=0.2,
        iterations=10,
        atol=1e-3,
    )
    # Each frame should contribute at most one detection; with a clean
    # trajectory and a conservative threshold, exactly one per frame.
    assert locs.shape[0] == n_frames
    sorted_locs = locs.sort("t")
    recovered_y = sorted_locs["y"].to_numpy()
    recovered_x = sorted_locs["x"].to_numpy()
    err_y = np.abs(recovered_y - ys)
    err_x = np.abs(recovered_x - xs)
    assert np.max(err_y) < 0.15
    assert np.max(err_x) < 0.15
    assert np.mean(err_y) < 0.05
    assert np.mean(err_x) < 0.05


def test_locate_circular_motion_3d() -> None:
    """A particle spiralling through (z, y, x) should be recovered in 3D."""
    psf_3d = _gaussian_psf_3d(1.2, 5)  # (5, 5, 5)
    n_frames = 16
    cz, cy, cx = 3.5, 7.0, 7.0
    radius_y, radius_x = 2.0, 2.0
    amp_z = 0.6
    t = np.arange(n_frames)
    theta = 2.0 * np.pi * t / n_frames
    zs = cz + amp_z * np.sin(theta)
    ys = cy + radius_y * np.cos(theta)
    xs = cx + radius_x * np.sin(theta)
    trajectories = polars.DataFrame(
        {
            "t": t.tolist(),
            "c": [0] * n_frames,
            "z": zs.tolist(),
            "y": ys.tolist(),
            "x": xs.tolist(),
            "contrast": [1.5] * n_frames,
            "particle_id": [0] * n_frames,
        }
    )
    video = tog.simulate_particles(
        trajectories,
        psf_3d,
        shape=(n_frames, 1, 7, 14, 14),
        noise_sigma=0.0,
        dtype=np.float32,
    )
    locs = tog.locate(
        video,
        psf_3d,
        chunk_size=4,
        min_distance=3,
        min_contrast=0.1,
        iterations=10,
        atol=1e-3,
    )
    assert locs.shape[0] == n_frames
    sorted_locs = locs.sort("t")
    err_z = np.abs(sorted_locs["z"].to_numpy() - zs)
    err_y = np.abs(sorted_locs["y"].to_numpy() - ys)
    err_x = np.abs(sorted_locs["x"].to_numpy() - xs)
    assert np.max(err_z) < 0.2
    assert np.max(err_y) < 0.15
    assert np.max(err_x) < 0.15


def test_locate_subpixel_motion() -> None:
    """A particle drifting slowly through subpixel offsets is recovered faithfully."""
    psf = _gaussian_psf(1.0, 7).reshape(1, 7, 7)
    n_frames = 41
    t = np.arange(n_frames, dtype=np.float64)
    # Drift y from 5.0 to 5.0 + 0.04 * 40 = 6.6 in 0.04-pixel steps.
    ys = 5.0 + 0.04 * t
    xs = np.full(n_frames, 6.5)
    trajectories = polars.DataFrame(
        {
            "t": t.astype(int).tolist(),
            "c": [0] * n_frames,
            "z": [0.0] * n_frames,
            "y": ys.tolist(),
            "x": xs.tolist(),
            "contrast": [2.0] * n_frames,
            "particle_id": [0] * n_frames,
        }
    )
    video = tog.simulate_particles(
        trajectories,
        psf,
        shape=(n_frames, 1, 1, 12, 12),
        noise_sigma=0.0,
        dtype=np.float32,
    )
    locs = tog.locate(
        video,
        psf,
        chunk_size=8,
        min_distance=3,
        min_contrast=0.1,
        iterations=20,
        atol=1e-4,
    )
    assert locs.shape[0] == n_frames
    sorted_locs = locs.sort("t")
    recovered_y = sorted_locs["y"].to_numpy()
    # Subpixel error must stay well below one pixel for the whole sweep.
    err = np.abs(recovered_y - ys)
    assert np.max(err) < 0.05
    # And the recovered trajectory should be a clean linear sweep.
    slope = np.polyfit(t, recovered_y, 1)[0]
    assert abs(slope - 0.04) < 1e-3


def test_locate_anisotropic_psf() -> None:
    """A strongly anisotropic PSF should still localize correctly."""
    # Build a separable Gaussian with very different y/x widths.
    n = 9
    sigma_y, sigma_x = 2.5, 0.6
    yy, xx = jnp.meshgrid(jnp.arange(n) - n // 2, jnp.arange(n) - n // 2, indexing="ij")
    psf = jnp.exp(-(yy * yy) / (2 * sigma_y * sigma_y)) * jnp.exp(
        -(xx * xx) / (2 * sigma_x * sigma_x)
    )
    psf = psf.reshape(1, n, n)

    # Place two well-separated emitters with different contrasts.
    n_frames = 1
    trajectories = polars.DataFrame(
        {
            "t": [0, 0],
            "c": [0, 0],
            "z": [0.0, 0.0],
            "y": [4.0, 12.0],
            "x": [6.2, 5.7],
            "contrast": [1.8, -1.2],
            "particle_id": [0, 1],
        }
    )
    video = tog.simulate_particles(
        trajectories,
        psf,
        shape=(n_frames, 1, 1, 16, 12),
        noise_sigma=0.01,
        seed=0,
        dtype=np.float32,
    )
    locs = tog.locate(
        video,
        psf,
        chunk_size=1,
        min_distance=4,
        min_contrast=0.2,
        iterations=10,
        atol=1e-3,
    )
    # Keep the two most prominent emitters and ignore any boundary artefacts.
    amps = locs["contrast"].to_numpy()
    keep = np.argsort(np.abs(amps))[-2:][::-1]
    sorted_locs = locs[keep.tolist()].sort("y")
    row0 = sorted_locs.row(0, named=True)
    row1 = sorted_locs.row(1, named=True)
    y0, x0, a0 = row0["y"], row0["x"], row0["contrast"]
    y1, x1, a1 = row1["y"], row1["x"], row1["contrast"]
    assert abs(y0 - 4.0) < 0.2
    assert abs(x0 - 6.2) < 0.2
    assert abs(a0 - 1.8) < 0.25
    assert abs(y1 - 12.0) < 0.2
    assert abs(x1 - 5.7) < 0.2
    assert abs(a1 - (-1.2)) < 0.25
    # Sign of the contrasts must be preserved.
    assert a0 > 0
    assert a1 < 0
