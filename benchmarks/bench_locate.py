"""Benchmark for `toolsandogh.locate` frames-per-second performance.

Run with:

    .venv/bin/python benchmarks/bench_locate.py

Prints timing and frames-per-second for a representative synthetic video.
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np
import polars
import xarray as xr

import toolsandogh as tog


def make_psf() -> jnp.ndarray:
    n = 9
    sigma = 1.5
    y, x = jnp.meshgrid(jnp.arange(n) - n // 2, jnp.arange(n) - n // 2, indexing="ij")
    return jnp.exp(-(y * y + x * x) / (2 * sigma * sigma)).reshape(1, n, n)


def make_video(n_frames: int, hw: int, psf: jnp.ndarray) -> xr.DataArray:
    cy, cx = hw / 2, hw / 2
    radius = hw / 8
    t = np.arange(n_frames)
    theta = 2.0 * np.pi * t / max(n_frames, 1)
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
    return tog.simulate_particles(
        trajectories,
        psf,
        shape=(n_frames, 1, 1, hw, hw),
        noise_sigma=0.02,
        seed=0,
        dtype=np.float32,
    )


def bench(
    n_frames: int, hw: int, chunk_size, min_contrast: float = 0.2, warmup: int = 4, repeats: int = 6
) -> float:
    psf = make_psf()
    video = make_video(n_frames, hw, psf)

    # Warmup (JIT compilation, etc.).
    for _ in range(warmup):
        tog.locate(
            video, psf, chunk_size=chunk_size, min_contrast=min_contrast, iterations=10, atol=1e-3
        )

    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        locs = tog.locate(
            video, psf, chunk_size=chunk_size, min_contrast=min_contrast, iterations=10, atol=1e-3
        )
        # Force materialization.
        _ = locs.height
        t1 = time.perf_counter()
        best = min(best, t1 - t0)
    fps = n_frames / best
    print(
        f"  n_frames={n_frames:4d} hw={hw:3d} chunk={str(chunk_size):>3} mc={min_contrast:.2f}  "
        f"best={best * 1e3:8.2f} ms  fps={fps:8.2f}  n_locs={locs.height}"
    )
    return fps


def main() -> None:
    jax.config.update("jax_enable_x64", False)
    print(f"jax devices: {jax.devices()}  backend: {jax.default_backend()}")
    print("Baseline locate() benchmark")
    for hw in (64, 128):
        for cs in (1, 8):
            bench(64, hw, cs, min_contrast=0.2)
    print("auto chunk_size (default):")
    for hw in (64, 128):
        bench(64, hw, "auto", min_contrast=0.2)
    print("Dense case (min_contrast=0):")
    bench(64, 64, 8, min_contrast=0.0)


if __name__ == "__main__":
    main()
