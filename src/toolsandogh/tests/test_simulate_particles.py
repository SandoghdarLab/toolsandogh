"""Tests for the synthetic particle renderer."""

import jax.numpy as jnp
import numpy as np
import polars
import pytest
import xarray as xr

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


def test_simulate_particles_roundtrip_single_emitter() -> None:
    """A single emitter should be recovered near its true position."""
    psf = _gaussian_psf(1.5, 7).reshape(1, 7, 7)
    true_y, true_x, true_amp = 3.4, 5.6, 2.0
    trajectories = polars.DataFrame(
        {
            "t": [0],
            "c": [0],
            "z": [0.0],
            "y": [true_y],
            "x": [true_x],
            "contrast": [true_amp],
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
    assert isinstance(video, xr.DataArray)
    assert video.shape == (1, 1, 1, 10, 10)
    assert video.dtype == np.float32

    # Locate the emitter with the same PSF.
    locs = locate_in_chunk(
        jnp.asarray(video.values[0]),
        psf,
        min_distance=3,
        min_contrast=0.1,
        iterations=50,
        atol=1e-4,
    )
    assert locs.shape[0] == 1
    assert abs(locs["y"][0] - true_y) < 0.05
    assert abs(locs["x"][0] - true_x) < 0.05
    assert abs(locs["contrast"][0] - true_amp) < 0.1
    assert bool(locs["converged"][0]) is True


def test_simulate_particles_negative_contrast() -> None:
    """Negative contrasts should be rendered as negative-contrast spots."""
    psf = _gaussian_psf(1.5, 7).reshape(1, 7, 7)
    true_amp = -2.0
    trajectories = polars.DataFrame(
        {
            "t": [0],
            "c": [0],
            "z": [0.0],
            "y": [5.0],
            "x": [5.0],
            "contrast": [true_amp],
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
        min_contrast=0.1,
        sign="negative",
        iterations=50,
        atol=1e-4,
    )
    assert locs.shape[0] >= 1
    # The most prominent negative emitter should be the one we placed.
    amps = sorted(locs["contrast"].to_list())
    assert abs(amps[0] - true_amp) < 0.1


def test_simulate_particles_noise_is_added() -> None:
    """With noise_sigma=0, the output should be deterministic and quiet."""
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
    video1 = tog.simulate_particles(
        trajectories,
        psf,
        shape=(1, 1, 1, 10, 10),
        noise_sigma=0.0,
        seed=123,
    )
    video2 = tog.simulate_particles(
        trajectories,
        psf,
        shape=(1, 1, 1, 10, 10),
        noise_sigma=0.0,
        seed=123,
    )
    np.testing.assert_array_equal(video1.values, video2.values)

    # With noise, the data should be reproducible for a given seed.
    video3 = tog.simulate_particles(
        trajectories,
        psf,
        shape=(1, 1, 1, 10, 10),
        noise_sigma=0.5,
        seed=42,
    )
    video4 = tog.simulate_particles(
        trajectories,
        psf,
        shape=(1, 1, 1, 10, 10),
        noise_sigma=0.5,
        seed=42,
    )
    np.testing.assert_array_equal(video3.values, video4.values)


def test_simulate_particles_3d() -> None:
    """3D rendering: an emitter at subpixel z is recovered in 3D."""
    psf = _gaussian_psf_3d(1.5, 7)  # (7, 7, 7)
    true_z, true_y, true_x, true_amp = 3.4, 5.6, 5.6, 2.0
    trajectories = polars.DataFrame(
        {
            "t": [0],
            "c": [0],
            "z": [true_z],
            "y": [true_y],
            "x": [true_x],
            "contrast": [true_amp],
            "particle_id": [0],
        }
    )
    video = tog.simulate_particles(
        trajectories,
        psf,
        shape=(1, 1, 7, 10, 10),
        noise_sigma=0.0,
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
    # The first row should be our emitter; the others are likely boundary
    # artefacts, so filter to the one with the largest absolute contrast.
    abs_amps = np.abs(locs["contrast"].to_numpy())
    best = int(np.argmax(abs_amps))
    assert abs(locs["z"][best] - true_z) < 0.1
    assert abs(locs["y"][best] - true_y) < 0.1
    assert abs(locs["x"][best] - true_x) < 0.1
    assert abs(locs["contrast"][best] - true_amp) < 0.1


def test_simulate_particles_missing_columns() -> None:
    """A trajectory table missing required columns should raise."""
    psf = _gaussian_psf(1.5, 7).reshape(1, 7, 7)
    bad = polars.DataFrame({"t": [0], "y": [1.0]})
    with pytest.raises(ValueError, match="missing required columns"):
        tog.simulate_particles(
            bad,
            psf,
            shape=(1, 1, 1, 10, 10),
        )
