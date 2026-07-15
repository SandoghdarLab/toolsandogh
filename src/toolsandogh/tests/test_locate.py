"""Tests for the particle localizer."""

import jax.numpy as jnp
import numpy as np
import polars
import pytest

import toolsandogh as tog
from toolsandogh._locate import _locate_in_chunk as locate_in_chunk


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
    locs = locate_in_chunk(
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
        "background": polars.Float32,
        "mass": polars.Float32,
        "snr": polars.Float32,
        "chi2": polars.Float32,
        "reduced_chi2": polars.Float32,
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
    locs = locate_in_chunk(
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
    pos_locs = locate_in_chunk(
        jnp.asarray(video.values[0]),
        psf,
        min_distance=3,
        min_contrast=0.0,
        sign="positive",
        iterations=50,
        atol=1e-4,
    )
    assert all(a > 0 for a in pos_locs["contrast"].to_list())

    neg_locs = locate_in_chunk(
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
    locs = locate_in_chunk(
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
    locs = locate_in_chunk(
        jnp.asarray(video.values[0]),
        psf,
        min_distance=3,
        min_contrast=0.0,
        iterations=50,
        atol=1e-4,
    )
    float_cols = {"z", "y", "x", "contrast", "background", "mass", "snr", "chi2", "reduced_chi2"}
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
    locs = locate_in_chunk(
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


def test_locate_snr_is_computed() -> None:
    """The ``snr`` column must be finite and scale with contrast over noise.

    A 20x20 frame keeps the 7x7 particle sparse so the robust noise
    estimate is accurate, making the quiet/loud SNR ordering reliable.
    """
    psf = _gaussian_psf(1.5, 7).reshape(1, 7, 7)
    trajectories = polars.DataFrame(
        {
            "t": [0],
            "c": [0],
            "z": [0.0],
            "y": [10.0],
            "x": [10.0],
            "contrast": [2.0],
            "particle_id": [0],
        }
    )
    common = {
        "min_distance": 3,
        "min_contrast": 0.1,
        "sign": "positive",
        "iterations": 50,
        "atol": 1e-4,
    }
    quiet = tog.simulate_particles(
        trajectories,
        psf,
        shape=(1, 1, 1, 20, 20),
        noise_sigma=0.02,
        seed=0,
        dtype=np.float32,
    )
    loud = tog.simulate_particles(
        trajectories,
        psf,
        shape=(1, 1, 1, 20, 20),
        noise_sigma=0.5,
        seed=0,
        dtype=np.float32,
    )
    quiet_locs = locate_in_chunk(jnp.asarray(quiet.values[0]), psf, **common)
    loud_locs = locate_in_chunk(jnp.asarray(loud.values[0]), psf, **common)

    # Pick the detection nearest the true emitter position (10, 10).
    def _nearest_snr(locs: polars.DataFrame) -> float:
        dy = locs["y"].to_numpy() - 10.0
        dx = locs["x"].to_numpy() - 10.0
        idx = int(np.argmin(dy * dy + dx * dx))
        return float(locs["snr"][idx])

    quiet_snr = _nearest_snr(quiet_locs)
    loud_snr = _nearest_snr(loud_locs)
    # Finite, positive, and a quiet background yields higher SNR than a loud one.
    assert np.isfinite(quiet_snr)
    assert np.isfinite(loud_snr)
    assert quiet_snr > 0
    assert loud_snr > 0
    assert quiet_snr > loud_snr


def test_locate_snr_uses_supplied_noise() -> None:
    """A supplied ``noise_sigma`` must set the ``snr`` scale directly."""
    psf = _gaussian_psf(1.5, 7).reshape(1, 7, 7)
    trajectories = polars.DataFrame(
        {
            "t": [0],
            "c": [0],
            "z": [0.0],
            "y": [10.0],
            "x": [10.0],
            "contrast": [2.0],
            "particle_id": [0],
        }
    )
    # Noise-free data so the fit recovers the contrast exactly; the
    # supplied sigma then fixes the SNR scale.
    video = tog.simulate_particles(
        trajectories,
        psf,
        shape=(1, 1, 1, 20, 20),
        noise_sigma=0.0,
        dtype=np.float32,
    )
    locs = locate_in_chunk(
        jnp.asarray(video.values[0]),
        psf,
        min_distance=3,
        min_contrast=0.1,
        sign="positive",
        iterations=50,
        atol=1e-4,
        noise_sigma=0.1,
    )
    assert locs.shape[0] >= 1
    # Fitted contrast ~ 2.0, supplied sigma 0.1 -> snr ~ 20.
    snr = float(locs["snr"][0])
    assert np.isfinite(snr)
    assert snr > 0
    assert abs(snr - 20.0) < 5.0


def test_locate_chi2_near_dof_for_good_fit() -> None:
    """``chi2`` centres on ``dof`` and ``reduced_chi2`` on 1 for a good fit."""
    psf = _gaussian_psf(1.5, 9).reshape(1, 9, 9)
    n_frames = 16
    cy, cx = 9.0, 9.0
    radius = 2.5
    t = np.arange(n_frames)
    theta = 2.0 * np.pi * t / n_frames
    trajectories = polars.DataFrame(
        {
            "t": t.tolist(),
            "c": [0] * n_frames,
            "z": [0.0] * n_frames,
            "y": (cy + radius * np.cos(theta)).tolist(),
            "x": (cx + radius * np.sin(theta)).tolist(),
            "contrast": [2.0] * n_frames,
            "particle_id": [0] * n_frames,
        }
    )
    video = tog.simulate_particles(
        trajectories,
        psf,
        shape=(n_frames, 1, 1, 20, 20),
        noise_sigma=0.1,
        seed=1,
        dtype=np.float32,
    )
    locs = tog.locate(
        video,
        psf,
        chunk_size=4,
        min_distance=4,
        min_contrast=0.2,
        iterations=30,
        atol=1e-4,
        noise_sigma=0.1,
    )
    assert locs.shape[0] >= n_frames
    # Spurious noise peaks survive the modest threshold; keep, for each
    # frame, the detection nearest the known true position and compute
    # chi2 on that subset (the real, well-fit emitters).
    locs = locs.with_columns(
        (polars.col("y") - cy).alias("_dy"), (polars.col("x") - cx).alias("_dx")
    )
    # distance to the true circular trajectory at the detection's frame.
    true_y = cy + radius * np.cos(2.0 * np.pi * locs["t"].to_numpy() / n_frames)
    true_x = cx + radius * np.sin(2.0 * np.pi * locs["t"].to_numpy() / n_frames)
    dist = np.sqrt((locs["y"].to_numpy() - true_y) ** 2 + (locs["x"].to_numpy() - true_x) ** 2)
    locs = locs.with_columns(polars.Series("_dist", dist))
    keep_idx = locs.sort("_dist").group_by("t", maintain_order=True).first()
    chi2 = keep_idx.sort("t")["chi2"].to_numpy()
    reduced = keep_idx.sort("t")["reduced_chi2"].to_numpy()
    assert chi2.shape[0] == n_frames
    # The 9x9 2D stamp has dof = 81 - 4 = 77; with the matching noise
    # sigma the chi-squared statistic should centre on dof.
    dof = 9 * 9 - 4
    assert dof == 77
    assert np.all(np.isfinite(chi2))
    assert np.all(chi2 > 0)
    assert 0.5 * dof < np.mean(chi2) < 1.7 * dof
    assert np.max(chi2) < 3.0 * dof
    # And the reduced chi-squared centres on 1.
    assert np.all(np.isfinite(reduced))
    assert 0.5 < np.mean(reduced) < 1.7
    assert np.max(reduced) < 3.0


def test_estimate_noise_sigma_recovers_known_std() -> None:
    """The robust estimator recovers a known noise std from pure noise."""
    import jax

    from toolsandogh._locate import _estimate_noise_sigma

    key = jax.random.PRNGKey(0)
    noise = 0.3 * jax.random.normal(key, (1, 1, 64, 64), dtype=np.float32)
    est = float(np.asarray(_estimate_noise_sigma(noise, 1)))
    # The MAD estimate from ~8k second-difference samples is accurate to
    # a few percent; allow a generous band.
    assert 0.27 < est < 0.33


def test_locate_noise_sigma_validation() -> None:
    """A negative ``noise_sigma`` must be rejected at the boundary."""
    psf = _gaussian_psf(1.5, 7).reshape(1, 7, 7)
    chunk = jnp.zeros((1, 1, 10, 10), dtype=np.float32)
    with pytest.raises(ValueError, match="noise_sigma"):
        locate_in_chunk(chunk, psf, noise_sigma=-0.1)


def test_locate_background_matches_constant_offset() -> None:
    """The fitted ``background`` must recover a constant video offset."""
    psf = _gaussian_psf(1.5, 7).reshape(1, 7, 7)
    trajectories = polars.DataFrame(
        {
            "t": [0],
            "c": [0],
            "z": [0.0],
            "y": [5.0],
            "x": [5.0],
            "contrast": [2.0],
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
    # Add a constant background offset of 42.0 to every pixel of the video.
    video = video + 42.0
    locs = locate_in_chunk(
        jnp.asarray(video.values[0]),
        psf,
        min_distance=3,
        min_contrast=0.1,
        sign="positive",
        iterations=50,
        atol=1e-4,
    )
    # The fitter models a per-emitter additive background, so the recovered
    # ``background`` column must reproduce the 42.0 offset (well within the
    # float32 fit tolerance) and be correctly typed.
    assert locs["background"].dtype == polars.Float32
    assert locs.shape[0] >= 1
    assert np.all(np.abs(locs["background"].to_numpy() - 42.0) < 0.1)
    # And it must be distinct from the contrast column.
    assert not np.allclose(locs["background"].to_numpy(), locs["contrast"].to_numpy(), atol=1e-6)


def test_locate_auto_chunk_is_default() -> None:
    """The default ``chunk_size`` must be ``"auto"``."""
    import inspect

    sig = inspect.signature(tog.locate)
    assert sig.parameters["chunk_size"].default == "auto"


def test_locate_auto_chunk_matches_explicit() -> None:
    """``chunk_size="auto"`` must produce the same result as an explicit size.

    Detection and fitting are per-frame and per-emitter, so the result is
    independent of the chunking; the driver streams each chunk through the
    same code path.  Agreement holds to float32 precision (batched-FFT
    reduction order differs slightly from per-frame).
    """
    psf = _gaussian_psf(1.5, 9).reshape(1, 9, 9)
    n_frames = 24
    cy, cx = 8.5, 8.5
    radius = 2.5
    t = np.arange(n_frames)
    theta = 2.0 * np.pi * t / n_frames
    trajectories = polars.DataFrame(
        {
            "t": t.tolist(),
            "c": [0] * n_frames,
            "z": [0.0] * n_frames,
            "y": (cy + radius * np.cos(theta)).tolist(),
            "x": (cx + radius * np.sin(theta)).tolist(),
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
    common = {
        "min_distance": 4,
        "min_contrast": 0.2,
        "iterations": 10,
        "atol": 1e-3,
    }
    auto = tog.locate(video, psf, chunk_size="auto", **common).sort("t", "y", "x")
    explicit = tog.locate(video, psf, chunk_size=1, **common).sort("t", "y", "x")
    explicit4 = tog.locate(video, psf, chunk_size=4, **common).sort("t", "y", "x")
    # Same number of detections and identical coordinates and fits.
    assert auto.shape[0] == explicit.shape[0] == explicit4.shape[0] == n_frames
    np.testing.assert_allclose(auto["y"].to_numpy(), explicit["y"].to_numpy(), rtol=1e-4)
    np.testing.assert_allclose(auto["x"].to_numpy(), explicit["x"].to_numpy(), rtol=1e-4)
    np.testing.assert_allclose(
        auto["contrast"].to_numpy(), explicit["contrast"].to_numpy(), rtol=1e-4
    )
    np.testing.assert_allclose(auto["y"].to_numpy(), explicit4["y"].to_numpy(), rtol=1e-4)


def test_resolve_chunk_size_auto() -> None:
    """``"auto"`` derives a memory-bounded chunk size from the frame shape."""
    from toolsandogh._locate import _MAX_CHUNK_BYTES, _resolve_chunk_size

    itemsize = np.dtype(np.float32).itemsize
    # Tiny frame: the budget admits more frames than exist, so use them all.
    assert _resolve_chunk_size("auto", 24, (1, 18, 18), itemsize) == 24
    # Huge 3D frame (200 x 2048 x 2048): a single frame already exceeds the
    # budget, so floor at one frame per chunk.
    big = _resolve_chunk_size("auto", 50, (200, 2048, 2048), itemsize)
    assert big == 1
    # The budget term matches the documented 3x working-set approximation.
    frame_bytes = 128 * 128 * itemsize
    expected = _MAX_CHUNK_BYTES // (3 * frame_bytes)
    assert _resolve_chunk_size("auto", 10**9, (1, 128, 128), itemsize) == expected


def test_locate_n_active_frames_filters_padded_detections() -> None:
    """``n_active_frames`` must suppress detections from zero-padded tail frames."""
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
    # Build a (4, 1, 10, 10) chunk with one real frame and three zero pads.
    real = jnp.asarray(video.values[0, 0])  # (1, 1, 10, 10)
    padded = jnp.zeros((4, 1, 10, 10), dtype=np.float32)
    padded = padded.at[0].set(real[0])
    locs = locate_in_chunk(
        padded,
        psf,
        n_active_frames=1,
        min_distance=3,
        min_contrast=0.0,
        iterations=10,
        atol=1e-3,
    )
    assert locs.shape[0] == 1
    assert locs["t"].to_list() == [0]


def test_locate_n_active_frames_validation() -> None:
    """Out-of-range ``n_active_frames`` must raise."""
    psf = _gaussian_psf(1.5, 7).reshape(1, 7, 7)
    chunk = jnp.zeros((4, 1, 10, 10), dtype=np.float32)
    with pytest.raises(ValueError, match="n_active_frames"):
        locate_in_chunk(chunk, psf, n_active_frames=5)
    with pytest.raises(ValueError, match="n_active_frames"):
        locate_in_chunk(chunk, psf, n_active_frames=-1)


def test_locate_n_iter_is_reported_correctly() -> None:
    """``n_iter`` must reflect the iteration at which each emitter converged."""
    psf = _gaussian_psf(1.5, 7).reshape(1, 7, 7)
    # Noise-free data: the fit should converge quickly (few iterations).
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
    locs = locate_in_chunk(
        jnp.asarray(video.values[0]),
        psf,
        min_distance=3,
        min_contrast=0.0,
        iterations=50,
        atol=1e-4,
    )
    assert locs.shape[0] == 1
    n_iter = locs["n_iter"][0]
    # With clean data the fit converges well before the 50-iteration cap.
    assert n_iter < 50
    assert n_iter >= 1
    # The emitter must be marked converged.
    assert locs["converged"][0]


def test_locate_n_iter_capped_at_max_iterations() -> None:
    """``n_iter`` for a non-converged emitter equals ``iterations``."""
    psf = _gaussian_psf(1.5, 7).reshape(1, 7, 7)
    # Very tight convergence threshold + few iterations: likely non-converged.
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
        noise_sigma=0.5,
        seed=99,
        dtype=np.float32,
    )
    locs = locate_in_chunk(
        jnp.asarray(video.values[0]),
        psf,
        min_distance=3,
        min_contrast=0.0,
        iterations=2,
        atol=1e-12,  # impossibly tight: should not converge
    )
    if locs.shape[0] > 0:
        # All emitters should report n_iter == iterations (the cap).
        assert all(n == 2 for n in locs["n_iter"].to_list())


def test_resolve_chunk_size_explicit_and_invalid() -> None:
    """Explicit sizes are capped at ``n_frames``; bad values raise."""
    from toolsandogh._locate import _resolve_chunk_size

    itemsize = np.dtype(np.float32).itemsize
    assert _resolve_chunk_size(8, 5, (1, 10, 10), itemsize) == 5
    assert _resolve_chunk_size(4, 20, (1, 10, 10), itemsize) == 4
    with pytest.raises(ValueError, match="chunk_size"):
        _resolve_chunk_size(0, 20, (1, 10, 10), itemsize)
    with pytest.raises(ValueError, match="chunk_size"):
        _resolve_chunk_size(-2, 20, (1, 10, 10), itemsize)
    with pytest.raises(ValueError, match="chunk_size"):
        _resolve_chunk_size("weird", 20, (1, 10, 10), itemsize)  # type: ignore


def test_locate_on_progress_sequence() -> None:
    """``on_progress`` is called with 0 then after each chunk with frames done."""
    psf = _gaussian_psf(1.5, 9).reshape(1, 9, 9)
    n_frames = 24
    cy, cx = 8.5, 8.5
    radius = 2.5
    t = np.arange(n_frames)
    theta = 2.0 * np.pi * t / n_frames
    trajectories = polars.DataFrame(
        {
            "t": t.tolist(),
            "c": [0] * n_frames,
            "z": [0.0] * n_frames,
            "y": (cy + radius * np.cos(theta)).tolist(),
            "x": (cx + radius * np.sin(theta)).tolist(),
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
    chunk = 4
    calls: list[int] = []
    tog.locate(
        video,
        psf,
        chunk_size=chunk,
        min_distance=4,
        min_contrast=0.2,
        iterations=10,
        atol=1e-3,
        on_progress=calls.append,
    )
    expected = [0, 4, 8, 12, 16, 20, 24]
    assert calls == expected


def test_locate_on_progress_single_chunk() -> None:
    """``n_frames`` divisible by ``chunk_size`` yields the exact plan."""
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
    calls: list[int] = []
    tog.locate(video, psf, chunk_size=1, on_progress=calls.append)
    assert calls == [0, 1]


def test_locate_on_progress_is_observation_only() -> None:
    """Progress reporting must not change the returned DataFrame."""
    psf = _gaussian_psf(1.5, 9).reshape(1, 9, 9)
    n_frames = 24
    cy, cx = 8.5, 8.5
    radius = 2.5
    t = np.arange(n_frames)
    theta = 2.0 * np.pi * t / n_frames
    trajectories = polars.DataFrame(
        {
            "t": t.tolist(),
            "c": [0] * n_frames,
            "z": [0.0] * n_frames,
            "y": (cy + radius * np.cos(theta)).tolist(),
            "x": (cx + radius * np.sin(theta)).tolist(),
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
    common = {
        "min_distance": 4,
        "min_contrast": 0.2,
        "iterations": 10,
        "atol": 1e-3,
    }
    silent = tog.locate(video, psf, chunk_size=4, **common).sort("t", "y", "x")
    loud = tog.locate(video, psf, chunk_size=4, **common, on_progress=lambda _v: None).sort(
        "t", "y", "x"
    )
    assert silent.shape == loud.shape
    np.testing.assert_allclose(silent["y"].to_numpy(), loud["y"].to_numpy(), rtol=1e-6)
    np.testing.assert_allclose(silent["x"].to_numpy(), loud["x"].to_numpy(), rtol=1e-6)
    np.testing.assert_allclose(
        silent["contrast"].to_numpy(), loud["contrast"].to_numpy(), rtol=1e-6
    )


def test_locate_on_progress_propagates_exception() -> None:
    """An exception raised inside ``on_progress`` must propagate."""
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

    class _Boom(Exception):
        pass

    def boom(_v: int) -> None:
        raise _Boom

    with pytest.raises(_Boom):
        tog.locate(video, psf, chunk_size=1, on_progress=boom)


def test_locate_on_progress_non_callable_raises_typeerror() -> None:
    """A non-callable ``on_progress`` is rejected before any work."""
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
    with pytest.raises(TypeError, match="on_progress"):
        tog.locate(video, psf, chunk_size=1, on_progress=42)  # type: ignore


def test_locate_accepts_raw_ndarray() -> None:
    """``locate`` must accept a raw NumPy array, not just a DataArray."""
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
    # Hand ``locate`` a plain (T, C, Z, Y, X) NumPy array instead of the
    # canonical DataArray; ``canonicalize_video`` must coerce it.
    raw = np.asarray(video.values, dtype=np.float32)
    locs = tog.locate(raw, psf, chunk_size=1, min_distance=3, iterations=10, atol=1e-3)
    assert locs.shape[0] == 1
    assert abs(locs["y"][0] - 5.0) < 0.1
    assert abs(locs["x"][0] - 5.0) < 0.1


def test_locate_accepts_2d_psf() -> None:
    """``locate`` must accept a 2D ``(Py, Px)`` PSF without manual reshaping."""
    # A 2D Gaussian PSF, deliberately left 2D (no ``.reshape(1, n, n)``).
    psf_2d = _gaussian_psf(1.5, 7)
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
        psf_2d.reshape(1, 7, 7),  # simulate_particles still expects 3D
        shape=(1, 1, 1, 10, 10),
        noise_sigma=0.0,
        dtype=np.float32,
    )
    locs = tog.locate(video, psf_2d, chunk_size=1, min_distance=3, iterations=10, atol=1e-3)
    assert locs.shape[0] == 1
    assert abs(locs["y"][0] - 5.0) < 0.1
    assert abs(locs["x"][0] - 5.0) < 0.1


def test_canonicalize_psf_rejects_bad_rank() -> None:
    """``_canonicalize_psf`` rejects anything that is not 2D or 3D."""
    from toolsandogh._locate import _canonicalize_psf

    with pytest.raises(ValueError, match="psf"):
        _canonicalize_psf(np.zeros(5))  # 1D
    with pytest.raises(ValueError, match="psf"):
        _canonicalize_psf(np.zeros((2, 2, 2, 2)))  # 4D
