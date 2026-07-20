"""Tests for the trajectory linker."""

import jax.numpy as jnp
import numpy as np
import polars
import pytest

import toolsandogh as tog


def _gaussian_psf(sigma: float, n: int) -> jnp.ndarray:
    """Build a 2D Gaussian PSF of shape (n, n)."""
    y, x = jnp.meshgrid(jnp.arange(n) - n // 2, jnp.arange(n) - n // 2, indexing="ij")
    return jnp.exp(-(y * y + x * x) / (2 * sigma * sigma))


def _simulate_and_locate(
    trajectories: polars.DataFrame,
    psf: jnp.ndarray,
    video_shape: tuple[int, int, int, int, int],
    **locate_kwargs,
) -> polars.DataFrame:
    """Render a video and run the localizer on it."""
    video = tog.simulate_particles(
        trajectories,
        psf,
        shape=video_shape,
        noise_sigma=0.0,
        dtype=np.float32,
    )
    return tog.locate(
        video,
        psf,
        min_distance=3,
        min_contrast=0.1,
        iterations=10,
        atol=1e-3,
        **locate_kwargs,
    )


def test_link_linear_motion_single_particle() -> None:
    """A single particle moving linearly is one trajectory."""
    psf = _gaussian_psf(1.0, 7).reshape(1, 7, 7)
    n_frames = 11
    t = np.arange(n_frames)
    ys = 4.0 + 0.1 * t
    xs = 5.0 + 0.1 * t
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
    locs = _simulate_and_locate(trajectories, psf, (n_frames, 1, 1, 10, 10))
    assert locs.shape[0] == n_frames
    linked = tog.link(locs, search_range_micrometers=1.0, memory=0)
    assert linked.shape[0] == n_frames
    assert "particle_id" in linked.columns
    assert linked["particle_id"].dtype == polars.Int32
    # Every detection should share the same particle id.
    assert linked["particle_id"].n_unique() == 1


def test_link_two_particles_separated() -> None:
    """Two well-separated particles get two distinct trajectories."""
    psf = _gaussian_psf(0.8, 5).reshape(1, 5, 5)
    n_frames = 8
    t = np.arange(n_frames)
    # Two particles on parallel paths that stay clearly separated.
    y0 = 3.0 + 0.05 * t
    x0 = 3.0 + 0.05 * t
    y1 = 7.0 - 0.05 * t
    x1 = 7.0 + 0.05 * t
    trajectories = polars.DataFrame(
        {
            "t": np.concatenate([t, t]).tolist(),
            "c": [0] * 2 * n_frames,
            "z": [0.0] * 2 * n_frames,
            "y": np.concatenate([y0, y1]).tolist(),
            "x": np.concatenate([x0, x1]).tolist(),
            "contrast": [2.0] * 2 * n_frames,
            "particle_id": [0] * n_frames + [1] * n_frames,
        }
    )
    locs = _simulate_and_locate(trajectories, psf, (n_frames, 1, 1, 14, 14))
    linked = tog.link(locs, search_range_micrometers=1.0, memory=0)
    # Both particles should keep their ids throughout the video.
    sorted_linked = linked.sort(["frame", "y"])
    pids_per_frame = []
    for tt in range(n_frames):
        rows = sorted_linked.filter(polars.col("frame") == tt)
        assert rows.shape[0] == 2
        pids_per_frame.append(rows["particle_id"].to_list())
    particle_0_ids = {p[0] for p in pids_per_frame}
    particle_1_ids = {p[1] for p in pids_per_frame}
    assert len(particle_0_ids) == 1
    assert len(particle_1_ids) == 1
    assert particle_0_ids != particle_1_ids


def test_link_gap_closing() -> None:
    """A particle that disappears briefly is linked across the gap."""
    psf = _gaussian_psf(1.0, 7).reshape(1, 7, 7)
    n_frames = 11
    t = np.arange(n_frames)
    ys = 5.0 + 0.1 * t
    xs = 5.0  # constant
    # Skip frames 4, 5, 6 to create a gap of 3 frames.
    keep_mask = (t < 4) | (t > 6)
    trajectories = polars.DataFrame(
        {
            "t": t[keep_mask].tolist(),
            "c": [0] * int(keep_mask.sum()),
            "z": [0.0] * int(keep_mask.sum()),
            "y": ys[keep_mask].tolist(),
            "x": xs.tolist() if False else [5.0] * int(keep_mask.sum()),
            "contrast": [2.0] * int(keep_mask.sum()),
            "particle_id": [0] * int(keep_mask.sum()),
        }
    )
    locs = _simulate_and_locate(trajectories, psf, (n_frames, 1, 1, 12, 12))
    # With memory=3, the gap of 3 frames (frames 4, 5, 6 skipped) should
    # be closed: all detections share the same id.
    linked = tog.link(locs, search_range_micrometers=1.0, memory=3)
    assert linked["particle_id"].n_unique() == 1

    # With memory=0, the same gap is fatal: the trajectory is split.
    linked_strict = tog.link(locs, search_range_micrometers=1.0, memory=0)
    assert linked_strict["particle_id"].n_unique() > 1


def test_link_adaptive_search_range() -> None:
    """Adaptive step extends the search range for missing tracks."""
    psf = _gaussian_psf(1.0, 7).reshape(1, 7, 7)
    n_frames = 11
    t = np.arange(n_frames)
    # A particle that moves 0.5 px per frame, but is missing in the
    # middle frames (so the search range must be expanded for it).
    ys = 5.0 + 0.5 * t
    keep_mask = (t < 3) | (t > 7)
    n_kept = int(keep_mask.sum())
    trajectories = polars.DataFrame(
        {
            "t": t[keep_mask].tolist(),
            "c": [0] * n_kept,
            "z": [0.0] * n_kept,
            "y": ys[keep_mask].tolist(),
            "x": [5.0] * n_kept,
            "contrast": [2.0] * n_kept,
            "particle_id": [0] * n_kept,
        }
    )
    # Sanity: confirm the keep mask removes a contiguous gap.
    assert (t < 3).sum() + (t > 7).sum() == n_kept
    locs = _simulate_and_locate(trajectories, psf, (n_frames, 1, 1, 12, 12))
    # Without adaptive step, the search range of 1.0 cannot bridge
    # the gap of 5 missing frames (the prediction is stale), so the
    # trajectory is split.  With adaptive step, the range expands to
    # bridge the gap.
    linked_adaptive = tog.link(
        locs,
        search_range_micrometers=1.0,
        memory=5,
        adaptive_step=2.0,
    )
    # The adaptive run should merge everything into one trajectory.
    assert linked_adaptive["particle_id"].n_unique() == 1


def test_link_multi_channel_independent() -> None:
    """Particles in different channels are tracked independently."""
    psf = _gaussian_psf(1.0, 7).reshape(1, 7, 7)
    n_frames = 6
    t = np.arange(n_frames)
    # One particle per channel at the same spatial position.
    y0 = 5.0 + 0.1 * t
    x0 = 5.0 + 0.1 * t
    trajectories = polars.DataFrame(
        {
            "t": np.concatenate([t, t]).tolist(),
            "c": [0] * n_frames + [1] * n_frames,
            "z": [0.0] * 2 * n_frames,
            "y": np.concatenate([y0, y0]).tolist(),
            "x": np.concatenate([x0, x0]).tolist(),
            "contrast": [2.0] * 2 * n_frames,
            "particle_id": [0] * n_frames + [0] * n_frames,
        }
    )
    video = tog.simulate_particles(
        trajectories,
        psf,
        shape=(n_frames, 2, 1, 12, 12),
        noise_sigma=0.0,
        dtype=np.float32,
    )
    # Locate each channel separately, then concatenate the per-channel
    # tables so we can link them.
    parts = []
    for ch in (0, 1):
        locs_ch = tog.locate(
            video,
            psf,
            channel=ch,
            min_distance=3,
            min_contrast=0.1,
            iterations=10,
            atol=1e-3,
        )
        parts.append(locs_ch)
    locs = polars.concat(parts, how="vertical_relaxed")
    # Each channel should contribute its own trajectory.
    linked = tog.link(locs, search_range_micrometers=1.0, memory=0)
    channel_0_ids = set(linked.filter(polars.col("channel") == 0)["particle_id"].to_list())
    channel_1_ids = set(linked.filter(polars.col("channel") == 1)["particle_id"].to_list())
    assert len(channel_0_ids) == 1
    assert len(channel_1_ids) == 1
    # And the two channels should have distinct ids.
    assert channel_0_ids.isdisjoint(channel_1_ids)


def test_link_empty_input() -> None:
    """An empty input returns an empty output with the right schema."""
    locs = polars.DataFrame(
        schema={
            "channel": polars.Int32,
            "frame": polars.Int32,
            "z": polars.Float64,
            "y": polars.Float64,
            "x": polars.Float64,
        }
    )
    out = tog.link(locs, search_range_micrometers=1.0)
    assert "particle_id" in out.columns
    assert out["particle_id"].dtype == polars.Int32
    assert out.shape[0] == 0


def test_link_schema_strictness() -> None:
    """The returned DataFrame has a ``particle_id`` column of type Int32."""
    psf = _gaussian_psf(1.0, 7).reshape(1, 7, 7)
    n_frames = 3
    t = np.arange(n_frames)
    trajectories = polars.DataFrame(
        {
            "t": t.tolist(),
            "c": [0] * n_frames,
            "z": [0.0] * n_frames,
            "y": [5.0] * n_frames,
            "x": [5.0] * n_frames,
            "contrast": [2.0] * n_frames,
            "particle_id": [0] * n_frames,
        }
    )
    locs = _simulate_and_locate(trajectories, psf, (n_frames, 1, 1, 10, 10))
    linked = tog.link(locs, search_range_micrometers=1.0, memory=0)
    assert "particle_id" in linked.columns
    assert linked["particle_id"].dtype == polars.Int32
    # The original columns are preserved.
    for col in locs.columns:
        assert col in linked.columns


def test_link_validates_input() -> None:
    """The linker must reject malformed input clearly."""
    psf = _gaussian_psf(1.0, 7).reshape(1, 7, 7)
    n_frames = 3
    t = np.arange(n_frames)
    trajectories = polars.DataFrame(
        {
            "t": t.tolist(),
            "c": [0] * n_frames,
            "z": [0.0] * n_frames,
            "y": [5.0] * n_frames,
            "x": [5.0] * n_frames,
            "contrast": [2.0] * n_frames,
            "particle_id": [0] * n_frames,
        }
    )
    locs = _simulate_and_locate(trajectories, psf, (n_frames, 1, 1, 10, 10))
    # Missing required column.
    bad = locs.drop("frame")
    with pytest.raises(ValueError, match="missing required columns"):
        tog.link(bad, search_range_micrometers=1.0)
    # Negative memory.
    with pytest.raises(ValueError, match="memory"):
        tog.link(locs, search_range_micrometers=1.0, memory=-1)
    # Non-positive search_range.
    with pytest.raises(ValueError, match="search_range_micrometers"):
        tog.link(locs, search_range_micrometers=0.0)
    # Negative adaptive_step.
    with pytest.raises(ValueError, match="adaptive_step"):
        tog.link(locs, search_range_micrometers=1.0, adaptive_step=-1.0)


def test_link_search_range_modes_mutually_exclusive() -> None:
    """The two search-range arguments are mutually exclusive."""
    psf = _gaussian_psf(1.0, 7).reshape(1, 7, 7)
    n_frames = 3
    t = np.arange(n_frames)
    trajectories = polars.DataFrame(
        {
            "t": t.tolist(),
            "c": [0] * n_frames,
            "z": [0.0] * n_frames,
            "y": [5.0] * n_frames,
            "x": [5.0] * n_frames,
            "contrast": [2.0] * n_frames,
            "particle_id": [0] * n_frames,
        }
    )
    locs = _simulate_and_locate(trajectories, psf, (n_frames, 1, 1, 10, 10))
    with pytest.raises(ValueError, match="mutually exclusive"):
        tog.link(locs, search_range_pixels=1.0, search_range_micrometers=1.0)
    with pytest.raises(ValueError, match="Exactly one"):
        tog.link(locs)


def test_link_pixel_mode_matches_micrometre_mode_at_unit_scale() -> None:
    """At unit pixel scale, pixel and micrometre modes give the same links."""
    psf = _gaussian_psf(1.0, 7).reshape(1, 7, 7)
    n_frames = 8
    t = np.arange(n_frames)
    ys = 4.0 + 0.1 * t
    xs = 5.0 + 0.1 * t
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
    locs = _simulate_and_locate(trajectories, psf, (n_frames, 1, 1, 10, 10))
    # ``simulate_particles`` uses dy=dx=1.0, so 1 px == 1 µm and the two
    # modes are numerically identical.
    linked_um = tog.link(locs, search_range_micrometers=1.0, memory=0)
    linked_px = tog.link(locs, search_range_pixels=1.0, memory=0)
    assert linked_um["particle_id"].to_list() == linked_px["particle_id"].to_list()


def test_link_per_axis_tuple_3d() -> None:
    """A 3-tuple search range is interpreted as ``(z, y, x)``."""
    psf = _gaussian_psf(1.0, 7).reshape(1, 7, 7)
    n_frames = 3
    t = np.arange(n_frames)
    # A particle that moves only along y, by 0.3 px/frame.
    trajectories = polars.DataFrame(
        {
            "t": t.tolist(),
            "c": [0] * n_frames,
            "z": [0.0] * n_frames,
            "y": (5.0 + 0.3 * t).tolist(),
            "x": [5.0] * n_frames,
            "contrast": [2.0] * n_frames,
            "particle_id": [0] * n_frames,
        }
    )
    locs = _simulate_and_locate(trajectories, psf, (n_frames, 1, 1, 10, 10))
    # A generous y range and a tight z/x range: since the particle does
    # not move in z or x, all frames link into one track.
    linked = tog.link(locs, search_range_micrometers=(0.1, 2.0, 0.1), memory=0)
    assert linked["particle_id"].n_unique() == 1


def test_link_per_axis_tuple_requires_z_column() -> None:
    """A 3-tuple search range requires the ``z`` column to be present."""
    psf = _gaussian_psf(1.0, 7).reshape(1, 7, 7)
    n_frames = 3
    t = np.arange(n_frames)
    trajectories = polars.DataFrame(
        {
            "t": t.tolist(),
            "c": [0] * n_frames,
            "z": [0.0] * n_frames,
            "y": [5.0] * n_frames,
            "x": [5.0] * n_frames,
            "contrast": [2.0] * n_frames,
            "particle_id": [0] * n_frames,
        }
    )
    locs = _simulate_and_locate(trajectories, psf, (n_frames, 1, 1, 10, 10))
    # Drop the z column to force the error.
    locs_no_z = locs.drop("z")
    with pytest.raises(ValueError, match="missing required columns"):
        tog.link(locs_no_z, search_range_micrometers=(1.0, 1.0, 1.0))


def test_link_rejects_2_tuple() -> None:
    """A 2-tuple search range is rejected; only scalar or 3-tuple is valid."""
    psf = _gaussian_psf(1.0, 7).reshape(1, 7, 7)
    n_frames = 3
    t = np.arange(n_frames)
    trajectories = polars.DataFrame(
        {
            "t": t.tolist(),
            "c": [0] * n_frames,
            "z": [0.0] * n_frames,
            "y": [5.0] * n_frames,
            "x": [5.0] * n_frames,
            "contrast": [2.0] * n_frames,
            "particle_id": [0] * n_frames,
        }
    )
    locs = _simulate_and_locate(trajectories, psf, (n_frames, 1, 1, 10, 10))
    with pytest.raises(ValueError, match="3-tuple"):
        tog.link(locs, search_range_micrometers=(1.0, 2.0))


def test_link_without_channel_column() -> None:
    """A table without a ``channel`` column is linked as a single channel."""
    psf = _gaussian_psf(1.0, 7).reshape(1, 7, 7)
    n_frames = 6
    t = np.arange(n_frames)
    ys = 4.0 + 0.1 * t
    xs = 5.0 + 0.1 * t
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
    locs = _simulate_and_locate(trajectories, psf, (n_frames, 1, 1, 10, 10))
    # Drop the channel column; the result should still link into one
    # trajectory and must not add a ``channel`` column.
    locs_no_channel = locs.drop("channel")
    linked = tog.link(locs_no_channel, search_range_micrometers=1.0, memory=0)
    assert "channel" not in linked.columns
    assert linked["particle_id"].n_unique() == 1
    # All input columns are preserved.
    for col in locs_no_channel.columns:
        assert col in linked.columns


def test_link_on_progress_reports_rows() -> None:
    """The progress callback reports cumulative rows processed."""
    psf = _gaussian_psf(1.0, 7).reshape(1, 7, 7)
    n_frames = 8
    t = np.arange(n_frames)
    ys = 4.0 + 0.1 * t
    xs = 5.0 + 0.1 * t
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
    locs = _simulate_and_locate(trajectories, psf, (n_frames, 1, 1, 10, 10))
    seen: list[int] = []
    tog.link(locs, search_range_micrometers=1.0, on_progress=seen.append)
    # The first call is 0; the last equals the row count.
    assert seen[0] == 0
    assert seen[-1] == locs.height
    # Values are monotonically non-decreasing.
    assert all(b >= a for a, b in zip(seen, seen[1:]))


def test_link_on_progress_is_observation_only() -> None:
    """The result is identical with or without a progress callback."""
    psf = _gaussian_psf(1.0, 7).reshape(1, 7, 7)
    n_frames = 6
    t = np.arange(n_frames)
    ys = 4.0 + 0.1 * t
    xs = 5.0 + 0.1 * t
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
    locs = _simulate_and_locate(trajectories, psf, (n_frames, 1, 1, 10, 10))
    plain = tog.link(locs, search_range_micrometers=1.0)
    with_cb = tog.link(locs, search_range_micrometers=1.0, on_progress=lambda _: None)
    assert plain["particle_id"].to_list() == with_cb["particle_id"].to_list()


def test_link_on_progress_non_callable_raises_typeerror() -> None:
    """A non-callable ``on_progress`` is rejected before any work."""
    psf = _gaussian_psf(1.0, 7).reshape(1, 7, 7)
    n_frames = 3
    t = np.arange(n_frames)
    trajectories = polars.DataFrame(
        {
            "t": t.tolist(),
            "c": [0] * n_frames,
            "z": [0.0] * n_frames,
            "y": [5.0] * n_frames,
            "x": [5.0] * n_frames,
            "contrast": [2.0] * n_frames,
            "particle_id": [0] * n_frames,
        }
    )
    locs = _simulate_and_locate(trajectories, psf, (n_frames, 1, 1, 10, 10))
    with pytest.raises(TypeError, match="on_progress"):
        tog.link(locs, search_range_micrometers=1.0, on_progress=42)  # type: ignore
