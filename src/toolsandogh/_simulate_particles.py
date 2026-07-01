"""
Render synthetic videos from a trajectory table and a PSF model.

The function :func:`simulate_particles` takes a Polars trajectory table and
a 3D point-spread function (PSF) and returns a canonical
``(T, C, Z, Y, X)`` :class:`xarray.DataArray` filled with rendered emitters
plus Gaussian noise.  It is intended as a companion to
:func:`toolsandogh.locate` for generating test data.
"""

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import polars
import xarray as xr
from jaxtyping import Array, Float

from ._canonicalize_video import canonicalize_video


def simulate_particles(
    trajectories: polars.DataFrame,
    psf: Float[Array, "Z Y X"],
    *,
    shape: tuple[int, int, int, int, int],
    noise_sigma: float = 1.0,
    seed: int = 0,
    dtype: npt.DTypeLike = np.float32,
) -> xr.DataArray:
    """
    Render a synthetic canonical video from a trajectory table.

    For every row of ``trajectories`` the PSF is placed on the
    corresponding frame and channel, centered at the rounded
    ``(y, x)`` position and shifted by the subpixel offsets
    ``(z - round(z))``, ``(y - round(y))``, ``(x - round(x))``.  The
    shifted PSF is multiplied by the row's ``contrast`` and added to
    the frame.  After all emitters are placed, i.i.d. Gaussian noise
    with standard deviation ``noise_sigma`` is added to every pixel.

    Parameters
    ----------
    trajectories : polars.DataFrame
        A Polars DataFrame describing the emitters.  Required columns:
        ``t``, ``c``, ``z``, ``y``, ``x``, ``contrast``, ``particle_id``.
        The columns ``z``, ``y``, ``x`` are floating point positions in
        pixel units; ``contrast`` is the per-emitter contrast in the
        same arbitrary units as the data; ``t`` and ``c`` are integer
        frame and channel indices; ``particle_id`` is an integer
        trajectory identifier.
    psf : jax.Array
        The 3D point-spread function model with shape
        ``(Pz, Py, Px)``.
    shape : tuple of int
        The desired output shape ``(T, C, Z, Y, X)`` of the canonical
        video.
    noise_sigma : float
        Standard deviation of the additive Gaussian noise.
    seed : int
        Seed for the noise generator.
    dtype : numpy dtype
        Dtype of the output video.

    Returns
    -------
    xarray.DataArray
        A canonical ``(T, C, Z, Y, X)`` video with the rendered
        emitters and added noise.
    """
    # Validate the trajectories schema.
    required = {"t", "c", "z", "y", "x", "contrast", "particle_id"}
    missing = required - set(trajectories.columns)
    if missing:
        raise ValueError(f"trajectories is missing required columns: {sorted(missing)}")

    # Validate shape and PSF.
    if len(shape) != 5:
        raise ValueError(f"`shape` must have five entries, got {shape}.")
    T, C, Z, Y, X = shape
    Pz, Py, Px = psf.shape
    if Pz < 1 or Py < 1 or Px < 1:
        raise ValueError(f"PSF must have positive shape, got {psf.shape}.")
    if any(s < 1 for s in (T, C, Z, Y, X)):
        raise ValueError(f"`shape` entries must be positive, got {shape}.")

    # Validate dtype
    np_dtype = np.dtype(dtype)

    # Cast the PSF to the requested dtype.
    psf_arr = jnp.asarray(psf, dtype=np_dtype)

    # Allocate the video.
    video = jnp.zeros((T, C, Z, Y, X), dtype=np_dtype)

    # Pre-compute the per-axis FFT frequencies for the subpixel shift.
    pad = max(Py, Px, Pz)
    padded_shape = (Pz + 2 * pad, Py + 2 * pad, Px + 2 * pad)
    kz = jnp.fft.fftfreq(padded_shape[0])[:, None, None]
    ky = jnp.fft.fftfreq(padded_shape[1])[None, :, None]
    kx = jnp.fft.fftfreq(padded_shape[2])[None, None, :]

    # Pre-pad the PSF with edge values for FFT-based subpixel shift.
    psf_padded = _edge_pad(psf_arr, pad)

    def render_one_emitter(
        video: Float[Array, "T C Z Y X"],
        t: int,
        c: int,
        z: float,
        y: float,
        x: float,
        contrast: float,
    ) -> Float[Array, "T C Z Y X"]:
        """Place a single shifted PSF on the (t, c) frame."""
        if not (0 <= t < T and 0 <= c < C):
            return video

        z_int = int(round(z))
        y_int = int(round(y))
        x_int = int(round(x))
        dz = z - z_int
        dy = y - y_int
        dx = x - x_int

        if not (0 <= z_int < Z and 0 <= y_int < Y and 0 <= x_int < X):
            return video

        # Subpixel shift via FFT with edge-extension padding.
        phase = jnp.exp(-2j * jnp.pi * (kz * dz + ky * dy + kx * dx))
        shifted = jnp.fft.ifftn(jnp.fft.fftn(psf_padded) * phase).real
        shifted = shifted[pad : pad + Pz, pad : pad + Py, pad : pad + Px]

        # Place the shifted PSF in the frame, with clipping at the
        # boundaries.
        z_start = z_int - Pz // 2
        y_start = y_int - Py // 2
        x_start = x_int - Px // 2

        z_lo = max(z_start, 0)
        y_lo = max(y_start, 0)
        x_lo = max(x_start, 0)
        z_hi = min(z_start + Pz, Z)
        y_hi = min(y_start + Py, Y)
        x_hi = min(x_start + Px, X)

        psf_z_lo = z_lo - z_start
        psf_y_lo = y_lo - y_start
        psf_x_lo = x_lo - x_start
        psf_z_hi = psf_z_lo + (z_hi - z_lo)
        psf_y_hi = psf_y_lo + (y_hi - y_lo)
        psf_x_hi = psf_x_lo + (x_hi - x_lo)

        patch = contrast * shifted[psf_z_lo:psf_z_hi, psf_y_lo:psf_y_hi, psf_x_lo:psf_x_hi]

        return video.at[t, c, z_lo:z_hi, y_lo:y_hi, x_lo:x_hi].add(patch)

    for row in trajectories.iter_rows(named=True):
        video = render_one_emitter(
            video,
            t=int(row["t"]),
            c=int(row["c"]),
            z=float(row["z"]),
            y=float(row["y"]),
            x=float(row["x"]),
            contrast=float(row["contrast"]),
        )

    # Add Gaussian noise.
    if noise_sigma > 0.0:
        key = jax.random.PRNGKey(seed)
        noise = noise_sigma * jax.random.normal(key, video.shape, dtype=np_dtype)
        video = video + noise

    # Wrap in a canonical DataArray.
    coords = {
        "T": np.arange(T, dtype=np.float64),
        "C": np.arange(C, dtype=np.float64),
        "Z": np.arange(Z, dtype=np.float64),
        "Y": np.arange(Y, dtype=np.float64),
        "X": np.arange(X, dtype=np.float64),
    }
    da = xr.DataArray(
        np.asarray(video, dtype=dtype),
        dims=("T", "C", "Z", "Y", "X"),
        coords=coords,
    )
    return canonicalize_video(
        da,
        T=T,
        C=C,
        Z=Z,
        Y=Y,
        X=X,
        dtype=dtype,
    )


def _edge_pad(
    arr: Float[Array, "Z Y X"],
    pad: int,
) -> Float[Array, "Z2 Y2 X2"]:
    """
    Pad ``arr`` with its edge values along every axis.

    Parameters
    ----------
    arr : jax.Array
        The 3D array to be padded.
    pad : int
        Number of edge-valued cells to add on each side of every axis.

    Returns
    -------
    jax.Array
        The padded array with shape ``(Z + 2*pad, Y + 2*pad, X + 2*pad)``.
    """
    Z, Y, X = arr.shape
    padded = jnp.zeros((Z + 2 * pad, Y + 2 * pad, X + 2 * pad), dtype=arr.dtype)
    padded = padded.at[pad : pad + Z, pad : pad + Y, pad : pad + X].set(arr)
    padded = padded.at[:pad, pad : pad + Y, pad : pad + X].set(arr[0:1, :, :])
    padded = padded.at[Z + pad :, pad : pad + Y, pad : pad + X].set(arr[-1:, :, :])
    padded = padded.at[pad : pad + Z, :pad, pad : pad + X].set(arr[:, 0:1, :])
    padded = padded.at[pad : pad + Z, Y + pad :, pad : pad + X].set(arr[:, -1:, :])
    padded = padded.at[pad : pad + Z, pad : pad + Y, :pad].set(arr[:, :, 0:1])
    padded = padded.at[pad : pad + Z, pad : pad + Y, X + pad :].set(arr[:, :, -1:])
    return padded
