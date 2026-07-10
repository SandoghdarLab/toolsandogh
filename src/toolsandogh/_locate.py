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

# Fixed, reproducible device-memory budget for a single chunk when
# ``chunk_size="auto"``.  The per-chunk working set is approximated as three
# times the frame size (data plus matched-filter scores plus the boolean
# peak mask); the auto chunk size is ``budget // (3 * frame_bytes)``.  The
# value is hard-coded (not probed from the device) so that runs are
# reproducible across machines.
_MAX_CHUNK_BYTES = 1 << 30  # 1 GiB


def locate(
    video: xr.DataArray,
    psf: npt.ArrayLike,
    *,
    channel: int | str | float | None = None,
    starting_frame: int = 0,
    chunk_size: int | Literal["auto"] = "auto",
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
    chunk_size : int or {"auto"}, optional
        Number of frames processed per chunk.  Larger chunks reduce
        dispatch and synchronization overhead; smaller chunks bound peak
        device memory.  If ``"auto"`` (the default), the size is derived
        from a fixed, reproducible device-memory budget so that the
        per-chunk working set (data plus matched-filter scores and the
        boolean peak mask) stays under roughly 1 GiB.
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
    frame_shape = (
        int(video.sizes["Z"]),
        int(video.sizes["Y"]),
        int(video.sizes["X"]),
    )
    chunk_axis = _resolve_chunk_size(chunk_size, n_frames, frame_shape, np.dtype(dtype).itemsize)
    video = video.chunk({"T": chunk_axis})

    # Walk the blocks and accumulate the per-chunk results.  Each chunk is
    # computed independently from the Dask array, so only one chunk's data
    # is held in memory at a time.
    psf_jax = jnp.asarray(psf_arr, dtype=dtype)
    results: list[polars.DataFrame] = []
    for start in range(0, n_frames, chunk_axis):
        end = min(start + chunk_axis, n_frames)
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

    # 1+2. Fused matched-filter score and non-maximum-suppression peak
    #      mask.  A single JIT kernel produces the boolean (B, Z, Y, X) peak
    #      mask for the whole chunk; one eager ``argwhere`` then yields the
    #      integer coordinates of every detection -- a single host sync per
    #      chunk instead of one per frame.
    is_peak = _peak_mask_batch(
        chunk,
        psf,
        min_distance=min_distance,
        min_contrast=min_contrast,
        sign=sign,
    )  # (B, Z, Y, X)

    coords = jnp.argwhere(is_peak)  # (n, 4) -> [b, z, y, x]
    # ``argwhere``'s output size is data-dependent, so reading the
    # shape forces one host sync.  Materialize the whole array at once
    # and slice in NumPy from here on -- no further device round-trips.
    coords_np = np.asarray(coords)
    n_emitters = int(coords_np.shape[0])
    if n_emitters == 0:
        return _empty_result(channel=channel)

    b_idx = np.ascontiguousarray(coords_np[:, 0], dtype=np.int32)
    rows_z = np.ascontiguousarray(coords_np[:, 1], dtype=np.int32)
    rows_y = np.ascontiguousarray(coords_np[:, 2], dtype=np.int32)
    rows_x = np.ascontiguousarray(coords_np[:, 3], dtype=np.int32)

    # 3. Batched stamp extraction: pad the whole chunk once, then read every
    #    stamp with a single advanced-indexing gather (one device dispatch for
    #    all emitters, instead of one eager pad+slice per emitter).
    stamps = _extract_stamps_batch(chunk, b_idx, rows_z, rows_y, rows_x, (Pz, Py, Px))

    fit = _fit_emitters_batch(
        stamps,
        psf,
        iterations=iterations,
        atol=atol,
    )

    # 4. Compose the output table.
    t_col = b_idx + np.int32(starting_frame)
    c_scalar, c_dtype = _channel_scalar_and_dtype(channel)

    # Pull every fitted scalar off the device in a single sync: stack the
    # per-emitter float outputs into one (k, n) array and read it back once.
    fit_stack = jnp.stack(
        [
            fit["z_offset"],
            fit["y_offset"],
            fit["x_offset"],
            fit["contrast"],
            fit["mass"],
            fit["chi2"],
        ]
    )  # (6, n)
    fit_np = np.asarray(fit_stack, dtype=np.float32)
    z_offsets, y_offsets, x_offsets = fit_np[0], fit_np[1], fit_np[2]
    contrast_col = np.ascontiguousarray(fit_np[3])
    mass_col = np.ascontiguousarray(fit_np[4])
    chi2_col = np.ascontiguousarray(fit_np[5])
    n_iter_col = np.full(n_emitters, np.int32(iterations), dtype=np.int32)
    converged_col = np.asarray(fit["converged"], dtype=bool)

    # Subpixel-refined positions: integer peak + subpixel offset.
    z_col = (rows_z.astype(np.float32) + z_offsets).astype(np.float32)
    y_col = (rows_y.astype(np.float32) + y_offsets).astype(np.float32)
    x_col = (rows_x.astype(np.float32) + x_offsets).astype(np.float32)

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
# Matched filter + non-maximum suppression (fused, batched over frames)
# ---------------------------------------------------------------------------


@jax.jit(static_argnames=["min_distance", "sign"])
def _peak_mask_batch(
    chunk: Float[Array, "B Z Y X"],
    psf: Float[Array, "Pz Py Px"],
    *,
    min_distance: int,
    min_contrast: float,
    sign: Literal["both", "positive", "negative"],
) -> Float[Array, "B Z Y X"]:
    """
    Compute the matched-filter score and the boolean peak mask for every
    frame in ``chunk`` in a single JIT kernel.

    The non-maximum-suppression window has extent ``1`` along the frame
    axis so that detections never bleed across frames; the spatial window
    is ``(2*min_distance+1)**3``, matching the previous per-frame logic.
    """
    scores = jax.vmap(lambda frame: jsignal.correlate(frame, psf, mode="same", method="fft"))(chunk)

    window = (1, 2 * min_distance + 1, 2 * min_distance + 1, 2 * min_distance + 1)
    strides = (1, 1, 1, 1)

    if sign == "positive":
        local_ext = jax.lax.reduce_window(scores, -jnp.inf, jax.lax.max, window, strides, "same")
        is_peak = (scores == local_ext) & (scores > min_contrast)
    elif sign == "negative":
        local_ext = jax.lax.reduce_window(scores, jnp.inf, jax.lax.min, window, strides, "same")
        is_peak = (scores == local_ext) & (scores < -min_contrast)
    else:  # "both"
        local_max = jax.lax.reduce_window(scores, -jnp.inf, jax.lax.max, window, strides, "same")
        local_min = jax.lax.reduce_window(scores, jnp.inf, jax.lax.min, window, strides, "same")
        is_pos = (scores == local_max) & (scores > min_contrast)
        is_neg = (scores == local_min) & (scores < -min_contrast)
        is_peak = is_pos | is_neg

    return is_peak


# ---------------------------------------------------------------------------
# Stamp extraction (batched)
# ---------------------------------------------------------------------------


def _extract_stamps_batch(
    chunk: Float[Array, "B Z Y X"],
    b_idx: npt.NDArray[np.int32],
    z_idx: npt.NDArray[np.int32],
    y_idx: npt.NDArray[np.int32],
    x_idx: npt.NDArray[np.int32],
    shape: tuple[int, int, int],
) -> Float[Array, "n Pz Py Px"]:
    """
    Extract every emitter stamp in a single batched gather.

    The whole ``chunk`` is edge-padded once along the spatial axes; each
    stamp of ``shape`` centered on ``(z, y, x)`` is then read out with a
    single advanced-indexing gather.  This replaces the previous Python
    loop that re-padded a full frame per emitter.

    The stamp's index ``(Pz//2, Py//2, Px//2)`` corresponds to the frame
    position ``(z, y, x)``; out-of-bounds positions are filled with the
    edge value of the frame.
    """
    Pz, Py, Px = shape
    half_z, half_y, half_x = Pz // 2, Py // 2, Px // 2
    n = b_idx.shape[0]

    # Pad the spatial axes of the whole chunk once.  Because
    # ``half_z == Pz // 2`` (etc.), a stamp centred on frame position
    # ``(z, y, x)`` starts at padded position ``(z, y, x)``.
    padded = jnp.pad(
        chunk,
        (
            (0, 0),
            (half_z, Pz - half_z - 1),
            (half_y, Py - half_y - 1),
            (half_x, Px - half_x - 1),
        ),
        mode="edge",
    )

    # Read every stamp with a single advanced-indexing gather.  The index
    # arrays are built in NumPy (cheap) so the only JAX dispatch here is the
    # gather itself -- no ``vmap`` tracing, which matters when a chunk holds
    # only a handful of emitters.  All indices stay in range because the
    # padding above guarantees ``padded`` is large enough along every axis.
    oz, oy, ox = np.mgrid[:Pz, :Py, :Px]
    b_g = np.broadcast_to(b_idx[:, None, None, None], (n, Pz, Py, Px))
    z_g = z_idx[:, None, None, None] + oz[None]
    y_g = y_idx[:, None, None, None] + oy[None]
    x_g = x_idx[:, None, None, None] + ox[None]
    return padded[jnp.asarray(b_g), jnp.asarray(z_g), jnp.asarray(y_g), jnp.asarray(x_g)]


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

    def residual(params: Float[Array, " n_params"]) -> Float[Array, " m"]:
        return (stamp - model(params)).ravel()

    def cost(params: Float[Array, " n_params"]) -> Float[Array, ""]:
        r = residual(params)
        return 0.5 * jnp.sum(r * r)

    # Fixed-length Levenberg--Marquardt scan.  Every emitter runs the
    # same number of iterations, which keeps the ``vmap`` over
    # emitters happy.
    #
    # The normal equations use the Gauss--Newton approximation
    # ``J^T J`` of the Hessian, with ``J = jacfwd(residual)``.  For only
    # 4--5 parameters, forward-mode AD over the residual is far cheaper
    # than ``jax.hessian``'s reverse-over-reverse pass, and Gauss--Newton
    # is the standard model for least-squares refinement of a PSF.
    lambda0 = jnp.array(1e-3, dtype=stamp.dtype)

    def lm_step(carry, _):
        params, lambda_, prev_cost, _prev_delta_max = carry

        r = residual(params)
        jac = jax.jacfwd(residual)(params)  # (m, n_params)
        jtj = jac.T @ jac  # (n_params, n_params)
        grad = jac.T @ r  # gradient of 0.5 * ||r||^2

        diag_h = jnp.diag(jtj)
        damped = jtj + lambda_ * jnp.diag(diag_h)
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


def _resolve_chunk_size(
    chunk_size: int | Literal["auto"],
    n_frames: int,
    frame_shape: tuple[int, int, int],
    itemsize: int,
) -> int:
    """
    Resolve ``chunk_size`` into a concrete number of frames per chunk.

    For ``"auto"`` the size is derived from the fixed, reproducible
    device-memory budget :data:`_MAX_CHUNK_BYTES`: the per-chunk working
    set is approximated as three times the frame size (data plus
    matched-filter scores plus the boolean peak mask), and the budget is
    divided by that cost.  The result is floored at ``1`` (a single frame
    is the smallest unit; ``Z`` is never split) and capped at
    ``n_frames``.

    Raises
    ------
    ValueError
        If ``chunk_size`` is neither a positive integer nor ``"auto"``.
    """
    if chunk_size == "auto":
        frame_bytes = int(np.prod(frame_shape)) * itemsize
        per_frame_working = 3 * frame_bytes
        return max(1, min(n_frames, _MAX_CHUNK_BYTES // per_frame_working))
    if not isinstance(chunk_size, (int, np.integer)) or chunk_size < 1:
        raise ValueError(f"`chunk_size` must be a positive int or 'auto', got {chunk_size!r}.")
    return min(int(chunk_size), n_frames)
