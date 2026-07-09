"""
Localization of point spread functions.

The :func:`locate` function is the public entry point: it accepts a canonical
Dask-backed ``(T, C, Z, Y, X)`` :class:`xarray.DataArray` and a PSF model and
returns a Polars DataFrame of detected emitters with fitted positions,
contrasts, and per-emitter statistics.  The auxiliary function
:func:`locate_in_chunk` operates on a single in-memory chunk of shape ``(B, Z,
Y, X)``.
"""

from typing import Callable, Literal

import jax
import jax.numpy as jnp
import jax.scipy.signal as jsignal
import numpy as np
import numpy.typing as npt
import polars
import polars.datatypes
import xarray as xr
from jaxtyping import Array, Float

from ._canonicalize_video import canonicalize_video

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def locate(
    video: xr.DataArray,
    psf: npt.ArrayLike,
    *,
    channel: int | str | float | None = None,
    starting_frame: int = 0,
    chunk_size: int = 1,
    min_distance: int = 3,
    min_contrast: float = 0.0,
    sign: Literal["both", "positive", "negative"] = "both",
    iterations: int = 10,
    atol: float = 1e-3,
    dtype: npt.DTypeLike = np.float32,
) -> polars.DataFrame:
    """
    Locate particles in every frame of a (T, C, Z, Y, X) video.

    In each frame, this function detects peaks of the matched-filter response
    and refines each peak's position and contrast by fitting a subpixel-shifted
    copy of the PSF to a stamp extracted from the data.

    Parameters
    ----------
    video : xarray.DataArray
        A canonical ``(T, C, Z, Y, X)`` :class:`xarray.DataArray`,
        typically Dask-backed.
    psf : array-like
        The 3D point-spread function model, shape ``(Pz, Py, Px)``.
    channel : int or str or float, optional
        The channel to localize.  Required when the video has more than
        one channel.  May be a label (anything xarray's ``.sel``
        accepts) or an integer index.  Defaults to the only channel
        when ``C == 1``.  Whatever value is supplied is written
        verbatim to the ``c`` column of the output, with a matching
        dtype.
    starting_frame : int
        Frame index of the first frame of the video.  Used to populate
        the ``t`` column when concatenating results from multiple
        videos.
    chunk_size : int
        Number of frames per Dask block.
    min_distance : int
        Minimum separation (in pixels) between two detected peaks.
        Pixels closer than ``min_distance`` to a stronger peak are
        suppressed.
    min_contrast : float
        Minimum absolute value of the matched-filter score for a peak
        to be reported.
    sign : {"both", "positive", "negative"}
        Whether to detect only positive peaks, only negative peaks, or
        both.
    iterations : int
        Number of Levenberg--Marquardt iterations to run for every
        emitter (fixed, so the per-emitter work is uniform and the
        whole batch is friendly to ``vmap``).
    atol : float
        Convergence threshold on the maximum absolute parameter
        update.  Emitters with a final parameter update smaller than
        ``atol`` are marked ``converged=True``.
    dtype : numpy dtype
        Dtype used for the data during localization.

    Returns
    -------
    polars.DataFrame
        A Polars DataFrame with the columns
        ``t, c, z, y, x, contrast, mass, snr, chi2, n_iter, converged``.
    """
    # Cast the PSF to the requested dtype and check its rank.
    dtype = np.dtype(dtype)
    psf_arr = np.asarray(psf, dtype=dtype)
    if psf_arr.ndim != 3:
        raise ValueError(f"`psf` must be a 3D array, got an array of shape {psf_arr.shape}.")

    # Validate the video.
    canonicalize_video(video)

    # Select the channel.  ``channel`` may be a label (string) or an
    # integer index.  We keep the original label so it can be written
    # to the ``c`` column of the output unchanged.
    n_channels = int(video.sizes["C"])
    c_label: int | str | float
    if n_channels == 1:
        if channel is not None and channel != 0:
            raise ValueError(f"`channel` is {channel!r} but the video has only one channel.")
        video = video.isel(C=0)
        c_label = 0
    else:
        if channel is None:
            raise ValueError(f"Video has {n_channels} channels; please specify `channel`.")
        try:
            video = video.sel(C=channel)
        except (KeyError, ValueError) as exc:
            raise ValueError(f"Channel {channel!r} is not present in the video.") from exc
        c_label = channel

    # Restrict to the chosen channel and chunk along the time axis.
    n_frames = int(video.sizes["T"])
    video = video.chunk({"T": min(chunk_size, n_frames)})

    # Walk the blocks and accumulate the per-chunk results.
    psf_jax = jnp.asarray(psf_arr, dtype=dtype)
    results: list[polars.DataFrame] = []
    for start in range(0, n_frames, chunk_size):
        end = min(start + chunk_size, n_frames)
        block: xr.DataArray = video.isel(T=slice(start, end))
        chunk = jnp.asarray(block.data.compute(), dtype=dtype)
        results.append(
            locate_in_chunk(
                chunk=chunk,
                psf=psf_jax,
                starting_frame=starting_frame + start,
                channel=c_label,
                min_distance=min_distance,
                min_contrast=min_contrast,
                sign=sign,
                iterations=iterations,
                atol=atol,
            )
        )

    if not results:
        return _empty_result(channel=c_label)

    return polars.concat(results, how="vertical_relaxed")


def locate_in_chunk(
    chunk: Float[Array, "B Z Y X"],
    psf: Float[Array, "Z Y X"],
    *,
    starting_frame: int = 0,
    channel: int | str | float = 0,
    min_distance: int = 3,
    min_contrast: float = 0.0,
    sign: Literal["both", "positive", "negative"] = "both",
    iterations: int = 10,
    atol: float = 1e-3,
) -> polars.DataFrame:
    """
    Locate particles in a single (B, Z, Y, X) chunk.

    Parameters
    ----------
    chunk : jax.Array
        A dense ``(B, Z, Y, X)`` array of image data.
    psf : jax.Array
        The 3D point-spread function model, shape ``(Pz, Py, Px)``.
    starting_frame : int
        Frame index of the first frame in the chunk.  The ``t`` column
        of the returned DataFrame runs over
        ``[starting_frame, starting_frame + B - 1]``.
    channel : int or str or float
        Channel label written to the ``c`` column of the output.  The
        column's dtype mirrors the type of this argument.
    min_distance : int
        Minimum separation (in pixels) between two detected peaks.
    min_contrast : float
        Minimum absolute value of the matched-filter score for a peak
        to be reported.
    sign : {"both", "positive", "negative"}
        Whether to detect only positive peaks, only negative peaks, or
        both.
    iterations : int
        Number of Levenberg--Marquardt iterations to run for every
        emitter (fixed, so the per-emitter work is uniform and the
        whole batch is friendly to ``vmap``).
    atol : float
        Convergence threshold on the maximum absolute parameter
        update.  Emitters with a final parameter update smaller than
        ``atol`` are marked ``converged=True``.

    Returns
    -------
    polars.DataFrame
        A Polars DataFrame with the columns
        ``t, c, z, y, x, contrast, mass, snr, chi2, n_iter, converged``.
    """
    chunk = jnp.asarray(chunk)
    psf = jnp.asarray(psf)
    if chunk.ndim != 4:
        raise ValueError(f"`chunk` must be 4D (B, Z, Y, X), got shape {chunk.shape}.")
    if psf.ndim != 3:
        raise ValueError(f"`psf` must be 3D, got shape {psf.shape}.")
    B, Z, Y, X = chunk.shape
    Pz, Py, Px = psf.shape
    if Py > Y or Px > X:
        raise ValueError(f"PSF Y/X shape ({Py}, {Px}) must not exceed data Y/X shape ({Y}, {X}).")
    if Pz > Z and Z > 1:
        raise ValueError(f"PSF Z size {Pz} must not exceed data Z size {Z}.")

    # 1. Matched-filter score for every frame.
    scores = _matched_filter_batch(psf, chunk)  # (B, Z, Y, X)

    # 2. Per-frame peak detection (eager Python loop over frames).
    rows_t: list[int] = []
    rows_z: list[int] = []
    rows_y: list[int] = []
    rows_x: list[int] = []
    rows_score: list[float] = []
    for b in range(B):
        coords, peak_scores = _detect_peaks_one_frame(
            scores[b],
            min_distance=min_distance,
            min_contrast=min_contrast,
            sign=sign,
        )
        for (zi, yi, xi), score in zip(coords, peak_scores):
            rows_t.append(b)
            rows_z.append(int(zi))
            rows_y.append(int(yi))
            rows_x.append(int(xi))
            rows_score.append(float(score))

    if not rows_t:
        return _empty_result(channel=channel)

    # 3. Fit each emitter.
    n_emitters = len(rows_t)
    stamps = jnp.stack(
        [
            _extract_stamp(chunk[b], rows_z[i], rows_y[i], rows_x[i], (Pz, Py, Px))
            for i, b in enumerate(rows_t)
        ]
    )

    fit = _fit_emitters_batch(
        stamps,
        psf,
        iterations=iterations,
        atol=atol,
    )

    # 4. Compose the output table.
    t_col = np.asarray(rows_t, dtype=np.int32) + np.int32(starting_frame)
    c_scalar, c_dtype = _channel_scalar_and_dtype(channel)

    # Subpixel-refined positions: integer peak + subpixel offset.
    z_offsets = np.asarray(fit["z_offset"], dtype=np.float32)
    y_offsets = np.asarray(fit["y_offset"], dtype=np.float32)
    x_offsets = np.asarray(fit["x_offset"], dtype=np.float32)
    z_col = (np.asarray(rows_z, dtype=np.float32) + z_offsets).astype(np.float32)
    y_col = (np.asarray(rows_y, dtype=np.float32) + y_offsets).astype(np.float32)
    x_col = (np.asarray(rows_x, dtype=np.float32) + x_offsets).astype(np.float32)

    contrast_col = np.asarray(fit["contrast"], dtype=np.float32)
    mass_col = np.asarray(fit["mass"], dtype=np.float32)
    chi2_col = np.asarray(fit["chi2"], dtype=np.float32)
    n_iter_col = np.full(n_emitters, np.int32(iterations), dtype=np.int32)
    converged_col = np.asarray(fit["converged"], dtype=bool)

    snr_col: list[float | None] = [None] * n_emitters

    return polars.DataFrame(
        {
            "t": t_col,
            "c": c_scalar,
            "z": z_col,
            "y": y_col,
            "x": x_col,
            "contrast": contrast_col,
            "mass": mass_col,
            "snr": polars.Series(snr_col, dtype=polars.Float32),
            "chi2": chi2_col,
            "n_iter": n_iter_col,
            "converged": converged_col,
        },
        schema={
            "t": polars.Int32,
            "c": c_dtype,
            "z": polars.Float32,
            "y": polars.Float32,
            "x": polars.Float32,
            "contrast": polars.Float32,
            "mass": polars.Float32,
            "snr": polars.Float32,
            "chi2": polars.Float32,
            "n_iter": polars.Int32,
            "converged": polars.Boolean,
        },
    )


# ---------------------------------------------------------------------------
# Matched filter
# ---------------------------------------------------------------------------


@jax.jit
def _matched_filter_batch(
    psf: Float[Array, "Z Y X"],
    chunk: Float[Array, "B Z Y X"],
) -> Float[Array, "B Z Y X"]:
    """Compute the matched-filter score for every frame in ``chunk``."""
    return jax.vmap(lambda frame: jsignal.correlate(frame, psf, mode="same", method="fft"))(chunk)


# ---------------------------------------------------------------------------
# Peak detection (non-maximum suppression)
# ---------------------------------------------------------------------------


def _detect_peaks_one_frame(
    score: Float[Array, "Z Y X"],
    *,
    min_distance: int,
    min_contrast: float,
    sign: Literal["both", "positive", "negative"],
) -> tuple[Float[Array, "n 3"], Float[Array, " n"]]:
    """
    Find local extrema in a single score map.

    Returns the integer coordinates of each peak and the corresponding
    matched-filter score at that pixel.

    This function is eager: ``jnp.argwhere`` returns a variable-size
    output that is incompatible with tracing, and the per-frame work
    is dominated by the matched filter and the Levenberg--Marquardt
    fit.
    """
    window = 2 * int(min_distance) + 1

    if sign == "positive":
        local_ext = jax.lax.reduce_window(
            score,
            -jnp.inf,
            jax.lax.max,
            (window, window, window),
            (1, 1, 1),
            "same",
        )
        is_peak = (score == local_ext) & (score > min_contrast)
    elif sign == "negative":
        local_ext = jax.lax.reduce_window(
            score,
            jnp.inf,
            jax.lax.min,
            (window, window, window),
            (1, 1, 1),
            "same",
        )
        is_peak = (score == local_ext) & (score < -min_contrast)
    else:  # "both"
        local_max = jax.lax.reduce_window(
            score,
            -jnp.inf,
            jax.lax.max,
            (window, window, window),
            (1, 1, 1),
            "same",
        )
        local_min = jax.lax.reduce_window(
            score,
            jnp.inf,
            jax.lax.min,
            (window, window, window),
            (1, 1, 1),
            "same",
        )
        is_pos = (score == local_max) & (score > min_contrast)
        is_neg = (score == local_min) & (score < -min_contrast)
        is_peak = is_pos | is_neg

    coords = jnp.argwhere(is_peak)  # (n, 3)
    scores = score[coords[:, 0], coords[:, 1], coords[:, 2]]
    return coords, scores


# ---------------------------------------------------------------------------
# Stamp extraction
# ---------------------------------------------------------------------------


def _extract_stamp(
    frame: Float[Array, "Z Y X"],
    z: int,
    y: int,
    x: int,
    shape: tuple[int, int, int],
) -> Float[Array, "Pz Py Px"]:
    """
    Extract a stamp of ``shape`` centered on ``(z, y, x)`` from ``frame``.

    Out-of-bounds positions are filled with the edge value of the
    frame.  The stamp's index ``(Pz//2, Py//2, Px//2)`` corresponds
    to the frame position ``(z, y, x)``.
    """
    Pz, Py, Px = shape
    half_z, half_y, half_x = Pz // 2, Py // 2, Px // 2

    # Pad the frame with edge values.
    padded = jnp.pad(
        frame,
        ((half_z, Pz - half_z - 1), (half_y, Py - half_y - 1), (half_x, Px - half_x - 1)),
        mode="edge",
    )

    # The stamp's centre (at index (Pz//2, Py//2, Px//2)) should land
    # on the padded position that corresponds to the frame position
    # (z, y, x).  Since the frame starts at padded position
    # (half_z, half_y, half_x), the frame position (z, y, x) sits at
    # padded (z + half_z, y + half_y, x + half_x).  The stamp's centre
    # is Pz//2 and Py//2 in from the top-left of the stamp, so the
    # top-left of the stamp is at:
    z_p = z + half_z - Pz // 2
    y_p = y + half_y - Py // 2
    x_p = x + half_x - Px // 2
    return jax.lax.dynamic_slice(padded, (z_p, y_p, x_p), (Pz, Py, Px))


# ---------------------------------------------------------------------------
# Levenberg--Marquardt refinement
# ---------------------------------------------------------------------------


@jax.jit(static_argnames=["iterations"])
def _fit_emitters_batch(
    stamps: Float[Array, "n Pz Py Px"],
    psf: Float[Array, "Pz Py Px"],
    *,
    iterations: int,
    atol: float,
) -> dict[str, jax.Array]:
    """
    Fit all emitters in parallel with Levenberg--Marquardt.

    All emitters run a fixed number of ``iterations`` steps so the
    per-emitter work is uniform and the whole batch maps cleanly
    through ``jax.vmap``.  Convergence is decided afterwards by
    comparing the final parameter update to ``atol``.

    Returns a dictionary of per-emitter outputs:
    ``contrast``, ``background``, ``z_offset``, ``y_offset``,
    ``x_offset``, ``converged``, ``chi2``, ``mass``.
    """
    is_3d = psf.shape[0] > 1
    fit_one = jax.vmap(
        lambda stamp: _fit_one_emitter(
            stamp,
            psf,
            iterations=iterations,
            atol=atol,
            is_3d=is_3d,
        )
    )
    return fit_one(stamps)


def _make_shifted_psf_3d(
    psf: Float[Array, "Pz Py Px"],
    pad: int,
) -> tuple[Callable, int]:
    """
    Build the closure that shifts a 3D PSF by ``(dz, dy, dx)``.

    Returns ``(shifted_psf, n_shift)`` where ``shifted_psf`` takes a
    full parameter vector ``[contrast, bg, dz, dy, dx]`` and returns
    the shifted PSF (contrast and bg ignored), and ``n_shift = 3``.
    """
    Pz, Py, Px = psf.shape
    psf_padded = _edge_pad_3d(psf, pad)
    kz = jnp.fft.fftfreq(Pz + 2 * pad)[:, None, None]
    ky = jnp.fft.fftfreq(Py + 2 * pad)[None, :, None]
    kx = jnp.fft.fftfreq(Px + 2 * pad)[None, None, :]
    psf_fft = jnp.fft.fftn(psf_padded)

    def shifted_psf(params: Float[Array, "5"]) -> Float[Array, "Pz Py Px"]:
        _, _, dz, dy, dx = params
        phase = jnp.exp(-2j * jnp.pi * (kz * dz + ky * dy + kx * dx))
        full = jnp.fft.ifftn(psf_fft * phase).real
        return full[pad : pad + Pz, pad : pad + Py, pad : pad + Px]

    return shifted_psf, 3


def _make_shifted_psf_2d(
    psf: Float[Array, "Pz Py Px"],
    pad: int,
) -> tuple[Callable, int]:
    """
    Build the closure that shifts a 2D (``Pz = 1``) PSF by ``(dy, dx)``.

    Returns ``(shifted_psf, n_shift)`` where ``shifted_psf`` takes a
    full parameter vector ``[contrast, bg, dy, dx]`` and returns the
    shifted PSF, and ``n_shift = 2``.
    """
    Py, Px = psf.shape[1], psf.shape[2]
    psf_padded = _edge_pad_2d(psf[0], pad)
    ky = jnp.fft.fftfreq(Py + 2 * pad)[:, None]
    kx = jnp.fft.fftfreq(Px + 2 * pad)[None, :]
    psf_fft = jnp.fft.fftn(psf_padded)

    def shifted_psf(params: Float[Array, "4"]) -> Float[Array, "Pz Py Px"]:
        _, _, dy, dx = params
        phase = jnp.exp(-2j * jnp.pi * (ky * dy + kx * dx))
        full = jnp.fft.ifftn(psf_fft * phase).real
        return full[pad : pad + Py, pad : pad + Px]

    return shifted_psf, 2


def _fit_one_emitter(
    stamp: Float[Array, "Pz Py Px"],
    psf: Float[Array, "Pz Py Px"],
    *,
    iterations: int,
    atol: float,
    is_3d: bool,
) -> dict[str, jax.Array]:
    """
    Refine a single emitter with Levenberg--Marquardt.

    Runs exactly ``iterations`` damped Gauss-Newton steps.  The
    damping factor ``lambda`` is updated each iteration: divided by
    two on accept, multiplied by two on reject.  The returned
    ``converged`` flag is set when the final parameter update is
    smaller than ``atol``.

    Two parameterizations are supported via the static ``is_3d``
    flag (so the JIT can fold the branching):

    - ``is_3d=True``: 5 parameters ``(contrast, background, dz, dy, dx)``.
    - ``is_3d=False``: 4 parameters ``(contrast, background, dy, dx)``.

    The 2D case avoids a spurious unconstrained ``dz`` direction
    (the PSF is constant along ``z`` when ``Pz = 1``, so the
    subpixel z-shift is meaningless and the Levenberg--Marquardt
    solver would otherwise pick up numerical noise).

    Internally the model is unified: the parameter vector always
    has the form ``[contrast, bg, *shift_params]``, and the
    ``shifted_psf`` closure unpacks the shift parameters itself.
    This lets the cost function, the Levenberg--Marquardt step,
    and the scan be shared between the 2D and 3D code paths.
    """
    Pz, Py, Px = psf.shape
    pad = max(Pz, Py, Px)

    if is_3d:
        shifted_psf, n_shift = _make_shifted_psf_3d(psf, pad)
    else:
        shifted_psf, n_shift = _make_shifted_psf_2d(psf, pad)

    # Unified model: ``contrast * shifted_psf(params) + bg``.  The
    # closure knows how to unpack the shift parameters from ``params``.
    def model(params: Float[Array, " n_params"]) -> Float[Array, "Pz Py Px"]:
        return params[0] * shifted_psf(params) + params[1]

    # Initial guesses: contrast from the stamp centre, background from
    # the stamp median, and zero subpixel shifts.
    amp0 = stamp[Pz // 2, Py // 2, Px // 2]
    bg0 = jnp.median(stamp)
    shift0 = jnp.zeros(n_shift, dtype=stamp.dtype)
    params0 = jnp.concatenate([jnp.array([amp0, bg0], dtype=stamp.dtype), shift0])

    def cost(params: Float[Array, " n_params"]) -> Float[Array, ""]:
        r = (stamp - model(params)).ravel()
        return 0.5 * jnp.sum(r * r)

    # Fixed-length Levenberg--Marquardt scan.  Every emitter runs the
    # same number of iterations, which keeps the ``vmap`` over
    # emitters happy.
    lambda0 = jnp.array(1e-3, dtype=stamp.dtype)

    def lm_step(carry, _):
        params, lambda_, prev_cost, _prev_delta_max = carry

        _, grad = jax.value_and_grad(cost)(params)
        hess = jax.hessian(cost)(params)

        diag_h = jnp.diag(hess)
        damped = hess + lambda_ * jnp.diag(diag_h)
        delta = jnp.linalg.solve(damped, -grad)
        new_params = params + delta
        new_cost = cost(new_params)

        accept = new_cost < prev_cost
        new_lambda = jnp.where(accept, lambda_ * 0.5, lambda_ * 2.0)
        new_params = jnp.where(accept, new_params, params)
        new_cost = jnp.where(accept, new_cost, prev_cost)

        delta_max = jnp.max(jnp.abs(delta))
        return (new_params, new_lambda, new_cost, delta_max), None

    init_carry = (
        params0,
        lambda0,
        cost(params0),
        jnp.array(jnp.inf, dtype=stamp.dtype),
    )
    (final_params, _, final_cost, final_delta_max), _ = jax.lax.scan(
        lm_step, init_carry, None, length=iterations
    )

    contrast = final_params[0]
    bg_fit = final_params[1]
    if is_3d:
        dz, dy, dx = final_params[2], final_params[3], final_params[4]
    else:
        dy, dx = final_params[2], final_params[3]
        dz = jnp.array(0.0, dtype=stamp.dtype)

    mass = jnp.sum(stamp)
    converged = final_delta_max < atol

    return {
        "contrast": contrast,
        "background": bg_fit,
        "z_offset": dz,
        "y_offset": dy,
        "x_offset": dx,
        "converged": converged,
        "chi2": 2.0 * final_cost,
        "mass": mass,
    }


def _edge_pad_2d(
    arr: Float[Array, "Y X"],
    pad: int,
) -> Float[Array, "Y2 X2"]:
    """Pad a 2D array with its edge values along every axis."""
    Y, X = arr.shape
    padded = jnp.zeros((Y + 2 * pad, X + 2 * pad), dtype=arr.dtype)
    padded = padded.at[pad : pad + Y, pad : pad + X].set(arr)
    padded = padded.at[:pad, pad : pad + X].set(arr[0:1, :])
    padded = padded.at[Y + pad :, pad : pad + X].set(arr[-1:, :])
    padded = padded.at[pad : pad + Y, :pad].set(arr[:, 0:1])
    padded = padded.at[pad : pad + Y, X + pad :].set(arr[:, -1:])
    return padded


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _edge_pad_3d(
    arr: Float[Array, "Z Y X"],
    pad: int,
) -> Float[Array, "Z2 Y2 X2"]:
    """Pad a 3D array with its edge values along every axis."""
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


def _channel_scalar_and_dtype(
    channel: int | str | float,
) -> tuple[int | str | float | np.integer | np.floating, polars.datatypes.DataTypeClass]:
    """
    Map a channel label to a ``(scalar, polars dtype)`` pair.

    Booleans are folded into the integer case (``bool`` is a subclass
    of ``int``).  Anything that is not ``int``, ``str`` or ``float``
    falls back to the polars ``Object`` dtype.
    """
    if isinstance(channel, (bool, np.bool_)):
        channel = int(channel)
    if isinstance(channel, (int, np.integer)):
        return np.int32(channel), polars.Int32
    if isinstance(channel, (str, np.str_)):
        return str(channel), polars.String
    if isinstance(channel, (float, np.floating)):
        return np.float32(channel), polars.Float32
    return channel, polars.Object


def _empty_result(channel: int | str | float = 0) -> polars.DataFrame:
    """Return an empty Polars DataFrame with the locate schema."""
    c_scalar, c_dtype = _channel_scalar_and_dtype(channel)
    return polars.DataFrame(
        schema={
            "t": polars.Int32,
            "c": c_dtype,
            "z": polars.Float32,
            "y": polars.Float32,
            "x": polars.Float32,
            "contrast": polars.Float32,
            "mass": polars.Float32,
            "snr": polars.Float32,
            "chi2": polars.Float32,
            "n_iter": polars.Int32,
            "converged": polars.Boolean,
        },
    ).with_columns(polars.lit(c_scalar).alias("c"))
