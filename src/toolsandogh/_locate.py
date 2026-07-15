"""
Localization of point spread functions.

The :func:`locate` function is the public entry point: it accepts any
array-like video (canonicalized internally to a Dask-backed
``(T, C, Z, Y, X)`` :class:`xarray.DataArray`) and a 2D or 3D PSF model,
and returns a Polars DataFrame of detected emitters with fitted positions,
contrasts, and per-emitter statistics.  The private helper
:func:`_locate_in_chunk` operates on a single in-memory chunk of shape
``(B, Z, Y, X)`` and expects well-formed JAX-array arguments.
"""

from typing import Callable, Literal

import jax
import jax.numpy as jnp
import jax.scipy.signal as jsignal
import jax.scipy.stats as jstats
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

# Fixed emitter-batch size for the Levenberg--Marquardt kernel.  Emitter
# batches are padded to the next power of two (capped at this value) so
# that the JIT-compiled fitting kernel specialises on a small set of
# shapes rather than one per distinct emitter count.  This bounds the
# number of JIT compilations to O(log max_emitters) per video
# configuration while keeping per-batch waste under 2x.  The cap also
# bounds the Jacobian memory (``batch * m * n_params * itemsize``).
_MAX_EMITTERS_PER_BATCH = 256


def locate(
    video: npt.ArrayLike,
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
    noise_sigma: float | None = None,
    dtype: npt.DTypeLike = np.float32,
    on_progress: Callable[[int], None] | None = None,
) -> polars.DataFrame:
    """
    Locate particles in every frame of a (T, C, Z, Y, X) video.

    In each frame, this function detects peaks of the matched-filter response
    and refines each peak's position and contrast by fitting a subpixel-shifted
    copy of the PSF to a stamp extracted from the data.

    Parameters
    ----------
    video : array-like or xarray.DataArray
        The video to localize.  Any object accepted by
        :func:`canonicalize_video` may be supplied: a raw NumPy array,
        a Dask array, a list, or an already-canonical
        ``(T, C, Z, Y, X)`` :class:`xarray.DataArray`.  The argument is
        canonicalized before any work is done.
    psf : array-like
        The point-spread function model, shape ``(Py, Px)`` for a 2D
        (widefield) PSF or ``(Pz, Py, Px)`` for a 3D PSF.  A 2D PSF is
        promoted to ``(1, Py, Px)`` internally.
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
    noise_sigma : float, optional
        Standard deviation of the per-pixel Gaussian noise.  When
        ``None`` (the default) it is estimated robustly from each chunk
        via the median absolute deviation of second differences.  The
        value normalises the ``chi2``, ``reduced_chi2`` and ``snr`` output
        columns; supplying a known value yields statistics that are
        comparable across runs and devices.  The default is ``None``.
    dtype : numpy dtype
        Dtype used for the data during localization.
    on_progress : callable, optional
        A callback invoked with the number of frames completed so far.
        It is called once with ``0`` before any frame is processed
        (also the only call when the video has no frames), and again
        after one or more frames have been processed.  The final value
        equals ``video.sizes["T"]``.  ``None`` (the default) disables
        progress reporting.

    Returns
    -------
    polars.DataFrame
        A Polars DataFrame with the columns
        ``t, c, z, y, x, contrast, background, mass, snr, chi2, reduced_chi2, n_iter, converged``.

    Notes
    -----
    The ``chi2`` column is the chi-squared statistic
    ``sum(residual**2) / noise_sigma**2``, which follows a chi-squared
    distribution with ``dof = Pz*Py*Px - n_params`` degrees of freedom
    under a correct model and Gaussian noise.  The ``reduced_chi2`` column
    is ``chi2 / dof`` and has expectation 1.  The ``snr`` column is the
    fitted contrast divided by ``noise_sigma``.  When ``noise_sigma`` is
    not supplied, it is estimated per chunk as described under the
    ``noise_sigma`` parameter.
    """
    # Canonicalize the inputs.  ``video`` may be any array-like (a raw
    # NumPy array, a Dask array, a list, or an already-canonical DataArray);
    # ``canonicalize_video`` coerces it into the canonical Dask-backed
    # ``(T, C, Z, Y, X)`` representation.  ``psf`` is promoted to 3D and cast
    # to the requested dtype.
    dtype = np.dtype(dtype)
    video = canonicalize_video(video)
    psf_arr = _canonicalize_psf(psf, dtype=dtype)

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

    # Validate an optional progress callback at the public boundary.
    if on_progress is not None and not callable(on_progress):
        raise TypeError(f"`on_progress` must be callable, got {type(on_progress).__name__}.")
    if on_progress is not None:
        on_progress(0)

    # Walk the blocks and accumulate the per-chunk results.  Each chunk is
    # computed independently from the Dask array, so only one chunk's data
    # is held in memory at a time.  The tail chunk is zero-padded to the
    # full ``chunk_axis`` size so that every chunk has the same shape; this
    # keeps the peak-detection and fitting kernels on a single JIT
    # specialization.  Padded frames are zero and cannot produce detections
    # (their matched-filter score is zero, which never exceeds
    # ``min_contrast >= 0``), and ``n_active_frames`` tells
    # ``locate_in_chunk`` how many frames are real.
    psf_jax = jnp.asarray(psf_arr, dtype=dtype)
    results: list[polars.DataFrame] = []
    for start in range(0, n_frames, chunk_axis):
        end = min(start + chunk_axis, n_frames)
        n_active = end - start
        block: xr.DataArray = video.isel(T=slice(start, end))
        chunk_np = np.asarray(block.data.compute(), dtype=dtype)
        if n_active < chunk_axis:
            chunk_np = np.pad(
                chunk_np,
                ((0, chunk_axis - n_active), (0, 0), (0, 0), (0, 0)),
                mode="constant",
            )
        chunk = jnp.asarray(chunk_np)
        result = _locate_in_chunk(
            chunk=chunk,
            psf=psf_jax,
            n_active_frames=n_active,
            starting_frame=starting_frame + start,
            channel=c_label,
            min_distance=min_distance,
            min_contrast=min_contrast,
            sign=sign,
            iterations=iterations,
            atol=atol,
            noise_sigma=noise_sigma,
        )
        results.append(result)
        if on_progress is not None:
            on_progress(end)

    if not results:
        return _empty_result(channel=c_label)

    return polars.concat(results, how="vertical_relaxed")


def _locate_in_chunk(
    chunk: Float[Array, "B Z Y X"],
    psf: Float[Array, "Z Y X"],
    *,
    starting_frame: int = 0,
    channel: int | str | float = 0,
    n_active_frames: int | None = None,
    min_distance: int = 3,
    min_contrast: float = 0.0,
    sign: Literal["both", "positive", "negative"] = "both",
    iterations: int = 10,
    atol: float = 1e-3,
    noise_sigma: float | None = None,
) -> polars.DataFrame:
    """
    Locate particles in a single ``(B, Z, Y, X)`` chunk.

    This is the in-memory, single-chunk primitive used by :func:`locate`.
    Both ``chunk`` and ``psf`` must already be well-formed JAX arrays:
    ``chunk`` is a dense 4D ``(B, Z, Y, X)`` array and ``psf`` is a 3D
    ``(Pz, Py, Px)`` array (a 2D PSF must have been promoted to ``(1, Py,
    Px)`` by the caller).  Input canonicalization and validation are the
    responsibility of :func:`locate`, not of this function.

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
    n_active_frames : int, optional
        Number of frames at the start of ``chunk`` that contain real
        data.  When ``None`` (the default), all ``B`` frames are
        processed.  When the chunk has been zero-padded to a fixed size
        (as :func:`locate` does for the tail chunk), supply the original
        frame count here so that detections from padded frames are
        discarded.
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
    noise_sigma : float, optional
        Standard deviation of the per-pixel Gaussian noise.  When
        ``None`` (the default) it is estimated robustly from the
        chunk's active frames via the median absolute deviation of
        second differences.  The value normalises the ``chi2``,
        ``reduced_chi2`` and ``snr`` output columns.  The default is
        ``None``.

    Returns
    -------
    polars.DataFrame
        A Polars DataFrame with the columns
        ``t, c, z, y, x, contrast, background, mass, snr, chi2, reduced_chi2, n_iter, converged``.

    Notes
    -----
    The ``chi2`` column is the chi-squared statistic
    ``sum(residual**2) / noise_sigma**2``, which follows a chi-squared
    distribution with ``dof = Pz*Py*Px - n_params`` degrees of freedom
    under a correct model and Gaussian noise.  The ``reduced_chi2``
    column is ``chi2 / dof`` and has expectation 1.  The ``snr`` column
    is the fitted contrast divided by ``noise_sigma``.
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
    if n_active_frames is None:
        n_active_frames = B
    elif not (0 <= n_active_frames <= B):
        raise ValueError(f"`n_active_frames` must be in [0, {B}], got {n_active_frames}.")
    if noise_sigma is not None and noise_sigma < 0:
        raise ValueError(f"`noise_sigma` must be non-negative, got {noise_sigma!r}.")

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
    # Discard any detections that fall in zero-padded tail frames.
    if n_active_frames < B:
        coords_np = coords_np[coords_np[:, 0] < n_active_frames]
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

    # Resolve the per-pixel noise standard deviation: either the value
    # supplied by the caller or a robust estimate from the chunk's active
    # frames.  It is kept as a device-resident scalar so no extra host sync
    # is needed before the batched fit.
    if noise_sigma is None:
        noise_sigma_arr = _estimate_noise_sigma(chunk, n_active_frames)
    else:
        noise_sigma_arr = jnp.asarray(noise_sigma, dtype=chunk.dtype)

    # 4. Batched Levenberg--Marquardt refinement.  Stamps are processed in
    #    sub-batches whose size is the next power of two (capped at
    #    ``_MAX_EMITTERS_PER_BATCH``) so that the JIT-compiled fitting kernel
    #    specialises on a small set of shapes rather than one per distinct
    #    emitter count.  All sub-batch dispatches are queued asynchronously;
    #    a single host sync at the end materialises every fitted scalar
    #    (plus a second sync for the convergence flags).
    batch_size = 1 << (n_emitters - 1).bit_length()  # next power of two
    batch_size = min(batch_size, _MAX_EMITTERS_PER_BATCH)
    fit_keys = [
        "z_offset",
        "y_offset",
        "x_offset",
        "contrast",
        "background",
        "mass",
        "chi2",
        "reduced_chi2",
        "snr",
    ]
    fit_stacks: list[jax.Array] = []
    converged_stacks: list[jax.Array] = []
    n_iter_stacks: list[jax.Array] = []
    for i in range(0, n_emitters, batch_size):
        sub_stamps = stamps[i : i + batch_size]
        n_sub = sub_stamps.shape[0]
        if n_sub < batch_size:
            sub_stamps = jnp.pad(
                sub_stamps,
                ((0, batch_size - n_sub), (0, 0), (0, 0), (0, 0)),
                mode="constant",
            )
        sub_fit = _fit_emitters_batch(
            sub_stamps,
            psf,
            iterations=iterations,
            atol=atol,
            noise_sigma=noise_sigma_arr,
        )
        sub_stack = jnp.stack([sub_fit[k] for k in fit_keys])  # (9, batch)
        fit_stacks.append(sub_stack[:, :n_sub])
        converged_stacks.append(sub_fit["converged"][:n_sub])
        n_iter_stacks.append(sub_fit["n_iter"][:n_sub])

    # 5. Compose the output table.
    t_col = b_idx + np.int32(starting_frame)
    c_scalar, c_dtype = _channel_scalar_and_dtype(channel)

    # Pull every fitted scalar off the device in a single sync: concatenate
    # all sub-batch results and read back once.
    fit_np = np.asarray(jnp.concatenate(fit_stacks, axis=1), dtype=np.float32)
    z_offsets, y_offsets, x_offsets = fit_np[0], fit_np[1], fit_np[2]
    contrast_col = np.ascontiguousarray(fit_np[3])
    background_col = np.ascontiguousarray(fit_np[4])
    mass_col = np.ascontiguousarray(fit_np[5])
    chi2_col = np.ascontiguousarray(fit_np[6])
    reduced_chi2_col = np.ascontiguousarray(fit_np[7])
    snr_col = np.ascontiguousarray(fit_np[8])
    n_iter_col = np.asarray(jnp.concatenate(n_iter_stacks), dtype=np.int32)
    converged_col = np.asarray(jnp.concatenate(converged_stacks), dtype=bool)

    # Subpixel-refined positions: integer peak + subpixel offset.
    z_col = (rows_z.astype(np.float32) + z_offsets).astype(np.float32)
    y_col = (rows_y.astype(np.float32) + y_offsets).astype(np.float32)
    x_col = (rows_x.astype(np.float32) + x_offsets).astype(np.float32)

    return polars.DataFrame(
        {
            "t": t_col,
            "c": c_scalar,
            "z": z_col,
            "y": y_col,
            "x": x_col,
            "contrast": contrast_col,
            "background": background_col,
            "mass": mass_col,
            "snr": snr_col,
            "chi2": chi2_col,
            "reduced_chi2": reduced_chi2_col,
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
            "background": polars.Float32,
            "mass": polars.Float32,
            "snr": polars.Float32,
            "chi2": polars.Float32,
            "reduced_chi2": polars.Float32,
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
    noise_sigma: jax.Array,
) -> dict[str, jax.Array]:
    """
    Fit all emitters in parallel with Levenberg--Marquardt.

    All emitters run a fixed number of ``iterations`` steps so the
    per-emitter work is uniform and the whole batch maps cleanly
    through ``jax.vmap``.  Convergence is decided afterwards by
    comparing the final parameter update to ``atol``.

    ``noise_sigma`` is the per-pixel noise standard deviation used to
    normalise each emitter's ``chi2``, ``reduced_chi2`` and ``snr``
    outputs.  It is a scalar shared across the batch and is passed to
    the per-emitter closure unbatched (``in_axes=None``) so it is not
    replicated.

    Returns a dictionary of per-emitter outputs:
    ``contrast``, ``background``, ``z_offset``, ``y_offset``,
    ``x_offset``, ``converged``, ``n_iter``, ``chi2``,
    ``reduced_chi2``, ``mass``, ``snr``.
    """
    shifted_psf, n_shift = _make_shifted_psf(psf)
    sigma = jnp.asarray(noise_sigma, dtype=stamps.dtype)
    fit_one = jax.vmap(
        lambda stamp, sig: _fit_one_emitter(
            stamp,
            shifted_psf=shifted_psf,
            n_shift=n_shift,
            iterations=iterations,
            atol=atol,
            noise_sigma=sig,
        ),
        in_axes=(0, None),
    )
    return fit_one(stamps, sigma)


def _make_shifted_psf(
    psf: Float[Array, "Pz Py Px"],
) -> tuple[Callable, int]:
    """
    Precompute the PSF's padded FFT and frequency grid.

    Returns ``(shifted_psf, n_shift)`` where ``shifted_psf`` is a closure
    that takes the full parameter vector ``[contrast, bg, *shifts]`` and
    returns the subpixel-shifted PSF (contrast and bg are ignored by the
    closure), and ``n_shift`` is the number of shift parameters (3 for
    3D, 2 for 2D).  The 2D path is selected when the PSF is a single
    plane (``Pz == 1``).

    Building the FFT and frequency grids here -- once, outside the
    per-emitter ``vmap`` -- avoids recomputing them for every emitter
    in the batch.  Only the per-emitter phase ramp ``exp(...)`` and the
    inverse FFT run inside the batched loop.
    """
    Pz, Py, Px = psf.shape
    pad = max(Pz, Py, Px)
    if Pz > 1:
        psf_padded = jnp.pad(psf, pad, mode="edge")
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

    # 2D: the PSF is constant along z (Pz == 1); only (dy, dx) are fitted.
    psf_padded = jnp.pad(psf[0], pad, mode="edge")
    ky = jnp.fft.fftfreq(Py + 2 * pad)[:, None]
    kx = jnp.fft.fftfreq(Px + 2 * pad)[None, :]
    psf_fft = jnp.fft.fftn(psf_padded)

    def shifted_psf(params: Float[Array, "4"]) -> Float[Array, "Pz Py Px"]:
        _, _, dy, dx = params
        phase = jnp.exp(-2j * jnp.pi * (ky * dy + kx * dx))
        full = jnp.fft.ifftn(psf_fft * phase).real
        return full[pad : pad + Py, pad : pad + Px]

    return shifted_psf, 2


def _lm_refine(
    params0: jax.Array,
    residual: Callable[[jax.Array], jax.Array],
    cost: Callable[[jax.Array], jax.Array],
    *,
    iterations: int,
    atol: float,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    Run a fixed-length damped Gauss-Newton (Levenberg--Marquardt) scan.

    Each step solves the Gauss--Newton normal equations
    ``(J^T J + lambda * diag(J^T J)) delta = -J^T r`` with
    ``J = jacfwd(residual)``; the damping ``lambda`` is halved on
    accept and doubled on reject.  Once an emitter's
    ``max(abs(delta))`` falls below ``atol`` it is marked converged and
    **frozen** -- its parameters, damping, and cost are held fixed for
    the remainder of the scan so later iterations cannot perturb a
    converged result.

    Returns ``(final_params, final_cost, converged, n_iter)`` where
    ``final_cost`` is ``0.5 * sum(residual(final_params)**2)`` and
    ``n_iter`` is the 1-indexed iteration at which the emitter first
    converged, or ``iterations`` if it never converged.
    """
    lambda0 = jnp.array(1e-3, dtype=params0.dtype)

    def lm_step(carry, _):
        params, lambda_, prev_cost, converged, n_iter = carry

        r = residual(params)
        jac = jax.jacfwd(residual)(params)  # (m, n_params)
        jtj = jac.T @ jac  # (n_params, n_params)
        grad = jac.T @ r  # gradient of 0.5 * ||r||^2

        diag_h = jnp.diag(jtj)
        damped = jtj + lambda_ * jnp.diag(diag_h)
        delta = jnp.linalg.solve(damped, -grad)
        candidate_params = params + delta
        candidate_cost = cost(candidate_params)

        accept = candidate_cost < prev_cost
        step_params = jnp.where(accept, candidate_params, params)
        step_lambda = jnp.where(accept, lambda_ * 0.5, lambda_ * 2.0)
        step_cost = jnp.where(accept, candidate_cost, prev_cost)
        delta_max = jnp.max(jnp.abs(delta))

        # Freeze already-converged emitters so later iterations cannot
        # perturb a converged result.
        new_params = jnp.where(converged, params, step_params)
        new_lambda = jnp.where(converged, lambda_, step_lambda)
        new_cost = jnp.where(converged, prev_cost, step_cost)

        newly_converged = (delta_max < atol) & ~converged
        new_converged = converged | newly_converged
        new_n_iter = jnp.where(converged, n_iter, n_iter + 1)

        return (new_params, new_lambda, new_cost, new_converged, new_n_iter), None

    init_carry = (
        params0,
        lambda0,
        cost(params0),
        jnp.array(False),
        jnp.array(0, dtype=jnp.int32),
    )
    (final_params, _, final_cost, final_converged, final_n_iter), _ = jax.lax.scan(
        lm_step, init_carry, None, length=iterations
    )
    return final_params, final_cost, final_converged, final_n_iter


def _fit_outputs(
    stamp: jax.Array,
    final_params: jax.Array,
    final_cost: jax.Array,
    converged: jax.Array,
    n_iter: jax.Array,
    *,
    n_shift: int,
    noise_sigma: jax.Array,
) -> dict[str, jax.Array]:
    """
    Unpack fitted parameters and compute the per-emitter output columns.

    The parameter vector has the form ``[contrast, bg, *shifts]`` with
    ``len(shifts) == n_shift`` (3 for 3D, 2 for 2D).  The 2D case carries
    a zero ``z_offset`` since the PSF is constant along ``z``.

    Goodness of fit and signal-to-noise are normalised by the per-pixel
    noise standard deviation ``noise_sigma``.  ``final_cost`` is
    ``0.5 * sum(residual**2)``, so the sum of squared residuals is
    ``2 * final_cost`` and no residual is re-evaluated here.

    - ``chi2`` is the chi-squared statistic ``ssr / sigma**2``, which
      follows a chi-squared distribution with ``dof = stamp.size -
      n_params`` degrees of freedom under a correct model and Gaussian
      noise.
    - ``reduced_chi2`` is ``chi2 / dof`` and has expectation 1.
    - ``snr`` is the fitted contrast divided by ``sigma``.

    ``sigma`` is floored at a precision-relative value so that noise-free
    data (``sigma == 0``, residual ``~ 0``) yields a near-zero ``chi2``
    and a large-but-finite ``snr`` instead of NaN/inf.
    """
    contrast = final_params[0]
    bg_fit = final_params[1]
    if n_shift == 3:
        dz, dy, dx = final_params[2], final_params[3], final_params[4]
    else:
        dy, dx = final_params[2], final_params[3]
        dz = jnp.array(0.0, dtype=stamp.dtype)

    mass = jnp.sum(stamp)

    ssr = 2.0 * final_cost
    n_params = n_shift + 2
    dof = jnp.maximum(
        jnp.asarray(stamp.size - n_params, dtype=stamp.dtype),
        jnp.asarray(1, dtype=stamp.dtype),
    )
    stamp_scale = jnp.maximum(jnp.max(jnp.abs(stamp)), jnp.asarray(1.0, dtype=stamp.dtype))
    sigma_floor = jnp.asarray(jnp.sqrt(jnp.finfo(stamp.dtype).eps), dtype=stamp.dtype) * stamp_scale
    sigma = jnp.maximum(noise_sigma, sigma_floor)
    chi2 = ssr / (sigma * sigma)
    reduced_chi2 = chi2 / dof
    snr = contrast / sigma

    return {
        "contrast": contrast,
        "background": bg_fit,
        "z_offset": dz,
        "y_offset": dy,
        "x_offset": dx,
        "converged": converged,
        "n_iter": n_iter,
        "chi2": chi2,
        "reduced_chi2": reduced_chi2,
        "mass": mass,
        "snr": snr,
    }


def _fit_one_emitter(
    stamp: jax.Array,
    *,
    shifted_psf: Callable,
    n_shift: int,
    iterations: int,
    atol: float,
    noise_sigma: jax.Array,
) -> dict[str, jax.Array]:
    """
    Refine a single emitter with Levenberg--Marquardt.

    Builds the unified model ``contrast * shifted_psf(params) + bg``
    (the ``shifted_psf`` closure unpacks the shift parameters itself),
    forms the residual and cost, computes initial guesses, then
    delegates the fixed-length LM scan to :func:`_lm_refine` and the
    output assembly to :func:`_fit_outputs`.

    Two parameterizations are supported, selected by ``n_shift`` (a
    Python int so the branching folds into JIT):

    - ``n_shift == 3``: 5 parameters ``(contrast, background, dz, dy, dx)``.
    - ``n_shift == 2``: 4 parameters ``(contrast, background, dy, dx)``.

    The 2D case avoids a spurious unconstrained ``dz`` direction (the
    PSF is constant along ``z`` when ``Pz = 1``, so the subpixel
    z-shift is meaningless and the solver would otherwise pick up
    numerical noise).
    """
    Pz, Py, Px = stamp.shape

    def model(params: jax.Array) -> jax.Array:
        return params[0] * shifted_psf(params) + params[1]

    def residual(params: jax.Array) -> jax.Array:
        return (stamp - model(params)).ravel()

    def cost(params: jax.Array) -> jax.Array:
        r = residual(params)
        return 0.5 * jnp.sum(r * r)

    # Initial guesses: contrast from the stamp centre, background from
    # the stamp median, and zero subpixel shifts.
    amp0 = stamp[Pz // 2, Py // 2, Px // 2]
    bg0 = jnp.median(stamp)
    shift0 = jnp.zeros(n_shift, dtype=stamp.dtype)
    params0 = jnp.concatenate([jnp.array([amp0, bg0], dtype=stamp.dtype), shift0])

    final_params, final_cost, converged, n_iter = _lm_refine(
        params0,
        residual,
        cost,
        iterations=iterations,
        atol=atol,
    )
    return _fit_outputs(
        stamp,
        final_params,
        final_cost,
        converged,
        n_iter,
        n_shift=n_shift,
        noise_sigma=noise_sigma,
    )


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _canonicalize_psf(
    psf: npt.ArrayLike,
    *,
    dtype: npt.DTypeLike | None = None,
) -> np.ndarray:
    """
    Coerce a PSF model to the canonical 3D ``(Pz, Py, Px)`` form.

    A 2D ``(Py, Px)`` PSF (widefield) is promoted to ``(1, Py, Px)``.
    Any other rank is rejected.  The result is cast to ``dtype`` when
    supplied, otherwise the input dtype is preserved.

    Parameters
    ----------
    psf : array-like
        The PSF model, 2D or 3D.
    dtype : numpy dtype, optional
        The dtype to cast the PSF to.

    Returns
    -------
    numpy.ndarray
        A 3D ``(Pz, Py, Px)`` array of the requested dtype.

    Raises
    ------
    ValueError
        If ``psf`` is neither 2D nor 3D.
    """
    arr = np.asarray(psf)
    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]
    if arr.ndim != 3:
        raise ValueError(f"`psf` must be a 2D or 3D array, got an array of shape {arr.shape}.")
    if dtype is not None:
        arr = arr.astype(dtype)
    return arr


def _estimate_noise_sigma(
    chunk: Float[Array, "B Z Y X"],
    n_active_frames: int,
) -> jax.Array:
    """
    Estimate the per-pixel noise standard deviation of ``chunk``.

    A second-difference filter (taps ``[1, -2, 1]``) cancels smooth
    signal and constant background, so its output is dominated by noise.
    Second differences are pooled over every spatial axis with at least
    three samples, and the median absolute deviation (MAD) -- a robust
    scale estimator tolerant of the sparse emitter pixels -- is
    converted to a standard deviation.

    Two derived (non hard-coded) constants appear:

    * ``hf_gain`` -- the L2 gain of the high-pass filter, computed as
      ``sqrt(sum(taps**2))``; for ``[1, -2, 1]`` this is ``sqrt(6)``.
    * ``mad_to_sigma`` -- the MAD-to-standard-deviation factor for a
      normal distribution, ``1 / norm.ppf(0.75)`` (about 1.4826).

    A tiny floor guards against noise-free data, where the estimate is
    exactly zero; the per-emitter precision floor in
    :func:`_fit_outputs` dominates in practice.
    """
    active = chunk[:n_active_frames]
    Z, Y, X = active.shape[1], active.shape[2], active.shape[3]
    taps = jnp.array([1.0, -2.0, 1.0], dtype=active.dtype)
    hf_gain = jnp.sqrt(jnp.sum(taps**2))
    mad_to_sigma = 1.0 / jstats.norm.ppf(0.75)
    diffs: list[jax.Array] = []
    if X >= 3:
        diffs.append((active[..., 2:] - 2 * active[..., 1:-1] + active[..., :-2]).reshape(-1))
    if Y >= 3:
        diffs.append(
            (active[:, :, 2:, :] - 2 * active[:, :, 1:-1, :] + active[:, :, :-2, :]).reshape(-1)
        )
    if Z >= 3:
        diffs.append(
            (active[:, 2:, :, :] - 2 * active[:, 1:-1, :, :] + active[:, :-2, :, :]).reshape(-1)
        )
    if not diffs:
        return jnp.asarray(jnp.finfo(active.dtype).tiny, dtype=active.dtype)
    d = jnp.concatenate(diffs)
    med = jnp.median(d)
    mad = jnp.median(jnp.abs(d - med))
    sigma = mad_to_sigma * mad / hf_gain
    floor = jnp.asarray(jnp.finfo(active.dtype).tiny, dtype=active.dtype)
    return jnp.maximum(sigma, floor)


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
            "background": polars.Float32,
            "mass": polars.Float32,
            "snr": polars.Float32,
            "chi2": polars.Float32,
            "reduced_chi2": polars.Float32,
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
