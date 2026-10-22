"""
Localization of point spread functions.

The :func:`locate` function is the public entry point: it accepts any
array-like video (canonicalized internally to a Dask-backed
``(T, C, Z, Y, X)`` :class:`xarray.DataArray`) and a 2D or 3D PSF model,
and returns a Polars DataFrame of detected emitters with physical
(micrometre/millisecond) and discrete (index-space) coordinates plus
fitted contrasts and per-emitter statistics.  The private helper
:func:`_locate_in_chunk` operates on a single in-memory chunk of shape
``(B, Z, Y, X)`` and emits only index-space coordinates; :func:`locate`
scales these into physical units using the video's coordinate axes.
"""

import functools
import itertools
from typing import Callable, Literal

import jax
import jax.numpy as jnp
import jax.scipy.signal as jsignal
import numpy as np
import numpy.typing as npt
import polars
import polars.datatypes
import scipy.signal
import xarray as xr
from jaxtyping import Array, Bool, Float, Int

from ._canonicalize_video import _axis_origin_and_step, canonicalize_video

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Fixed, reproducible device-memory budget for a single chunk when
# ``chunk_size="auto"``.  The per-chunk working set is approximated as three
# times the frame size (data plus matched-filter scores plus the boolean
# peak mask); the auto chunk size is ``budget // (3 * frame_bytes)``.  The
# value is hard-coded (not probed from the device) so that runs are
# reproducible across machines.
_MAX_CHUNK_BYTES = 250_000_000

# Fixed emitter-batch size for the Levenberg--Marquardt kernel.  Emitter
# batches are padded to the next power of two (capped at this value) so
# that the JIT-compiled fitting kernel specialises on a small set of
# shapes rather than one per distinct emitter count.  This bounds the
# number of JIT compilations to O(log max_emitters) per video
# configuration while keeping per-batch waste under 2x.  The cap also
# bounds the Jacobian memory (``batch * m * n_params * itemsize``).
_MAX_EMITTERS_PER_BATCH = 256

# Bound (in voxels, per axis) on the subpixel shift fitted around each
# detected peak.  The peak is the integer maximum of the matched-filter
# score, so the true offset lies within about half a voxel; the bound
# leaves some slack for noise while preventing the fit from wandering
# off to (or past) the edge of the stamp, where the Fourier shift wraps
# around.  Emitters whose fit ends at the bound are reported with
# ``converged=False``.
_MAX_SUBPIXEL_SHIFT = 1.0

# Safety factor applied to the simulated worst-case side-lobe mismatch
# (see :func:`_sidelobe_tolerance`).  It covers effects the simulation does
# not model exactly, such as subpixel offsets between the simulated grid
# points and superposition of several emitters' side lobes.
_SIDELOBE_MARGIN = 2.0


def locate(
    video: npt.ArrayLike,
    psf: npt.ArrayLike,
    *,
    channel: int | str | float | None = None,
    chunk_size: int | Literal["auto"] = "auto",
    min_distance: int = 3,
    min_contrast: float = 0.5,
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
    copy of the PSF to a stamp extracted from the data.  Positions are reported
    in both physical units (micrometres, milliseconds) and index space
    (subpixel voxel/frame indices).

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
        promoted to ``(1, Py, Px)`` internally.  ``Pz`` must not exceed
        the video's ``Z`` size; in particular, 2D (``Z == 1``) videos
        require a 2D PSF.  The PSF is
        mean-subtracted and L2-normalized internally for the matched
        filter (so the detection score is independent of the PSF's DC
        level and pixel discretization), but the reported ``contrast``
        and ``background`` are rescaled back into the units of the
        supplied PSF: ``contrast`` is the emitter's amplitude (peak
        value above background) and ``background`` is the true additive
        offset.  A constant or near-constant PSF (zero mean-subtracted
        norm) is rejected, since no amplitude scale is defined for it.
        The PSF's centre is assumed to lie at index ``P // 2`` along
        every axis, and its window should be large enough for the PSF to
        have decayed to the background at the border: side lobes of very
        bright emitters (relative to ``min_contrast``) are only reliably
        suppressed if the PSF is not truncated.
    channel : int or str or float, optional
        The channel to localize, given as a coordinate label (anything
        xarray's ``.sel`` accepts).  Required when the video has more
        than one channel.  Defaults to the only channel when ``C == 1``.
        The value is written verbatim to the ``channel`` column of the
        output.
    chunk_size : int or {"auto"}, optional
        Number of frames processed per chunk.  Larger chunks reduce
        dispatch and synchronization overhead; smaller chunks bound peak
        device memory.  If ``"auto"`` (the default), the size is derived
        from a fixed, reproducible device-memory budget so that the
        per-chunk working set (data plus matched-filter scores and the
        boolean peak mask) stays under roughly 250 MB.
    min_distance : int
        Minimum separation (in pixels) between two detected peaks of the
        same sign.  A pixel is only a peak if it is the extremum of the
        ``(2*min_distance+1)``-wide cube around it; among exactly tied
        extrema, only the first in raster order is kept.
    min_contrast : float
        Minimum absolute amplitude for a peak to be reported, in the
        same units as the supplied PSF (i.e. a direct threshold on the
        ``contrast`` column, which holds the fitted amplitude).  Must
        be strictly positive; the default is ``0.5``.
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
        ``atol`` are marked ``converged=True``, unless their subpixel
        shift ended at its bound (see Notes).
    noise_sigma : float, optional
        Standard deviation of the per-pixel Gaussian noise.  When
        ``None`` (the default) it is estimated **per frame** from the
        standard deviation of second differences (a single-pass
        Laplacian-based noise estimator).  The value normalises the
        ``chi2``, ``reduced_chi2`` and ``snr`` output columns; supplying
        a known value yields statistics that are comparable across runs
        and devices.  The default is ``None``.
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
        ``t, channel, z, y, x, frame, slice, row, column, contrast,
        background, snr, chi2, reduced_chi2, n_iter, converged``.
        The ``t``, ``z``, ``y``, ``x`` columns are physical coordinates
        (float64, milliseconds and micrometres); the ``frame``/``slice``/
        ``row``/``column`` columns are index-space positions (``frame``
        is Int32, ``slice``/``row``/``column`` are Float32 with subpixel
        refinement); ``channel`` is the channel label.

    Notes
    -----
    Internally the PSF is mean-subtracted and L2-normalized to form the
    matched-filter kernel, but the results are reported in the units of
    the supplied PSF.  The fitted model is ``A * psf + B``; the
    ``contrast`` column holds the amplitude ``A`` (the peak value of the
    emitter above background) and the ``background`` column holds the
    true additive offset ``B``.  This makes ``min_contrast`` a direct
    threshold on the reported amplitude and keeps the output
    independent of how the PSF was discretized (window size, sampling).

    Detection runs the matched filter on edge-padded frames, so a
    constant background cancels up to the frame border.  Because the
    mean-subtracted PSF has side lobes, every emitter also produces
    weaker opposite-sign (and, for ringing PSFs, same-sign) responses
    around it.  Candidate peaks are therefore resolved greedily,
    strongest first, and only kept if their response, after subtracting
    the side lobes predicted from all stronger accepted emitters, still
    exceeds ``min_contrast`` plus a PSF-derived tolerance proportional
    to those emitters' strength.  Faint emitters close to much brighter
    ones may consequently be missed.

    The subpixel shift fitted around each detected peak is limited to
    one voxel per axis.  Emitters whose fit ends at that limit are
    reported with ``converged=False``.

    The ``chi2`` column is the chi-squared statistic
    ``sum(residual**2) / noise_sigma**2``, which follows a chi-squared
    distribution with ``dof = Pz*Py*Px - n_params`` degrees of freedom
    under a correct model and Gaussian noise.  The ``reduced_chi2`` column
    is ``chi2 / dof`` and has expectation 1.  The ``snr`` column is the
    fitted contrast (amplitude) divided by ``noise_sigma``.  When
    ``noise_sigma`` is not supplied, it is estimated per frame as
    described under the ``noise_sigma`` parameter.
    """
    # Canonicalize the inputs.  ``video`` may be any array-like (a raw
    # NumPy array, a Dask array, a list, or an already-canonical DataArray);
    # ``canonicalize_video`` coerces it into the canonical Dask-backed
    # ``(T, C, Z, Y, X)`` representation.  ``psf`` is promoted to 3D and cast
    # to the requested dtype.
    dtype = np.dtype(dtype)

    # Validate scalar parameters up front so we fail before any work.
    if min_contrast <= 0:
        raise ValueError(f"`min_contrast` must be strictly positive, got {min_contrast!r}.")
    if min_distance < 0:
        raise ValueError(f"`min_distance` must be non-negative, got {min_distance!r}.")
    if iterations < 0:
        raise ValueError(f"`iterations` must be non-negative, got {iterations!r}.")

    video = canonicalize_video(video)
    psf_arr = _canonicalize_psf(psf, dtype=dtype)

    # The detection/fitting kernels operate on the mean-subtracted,
    # L2-normalized PSF (so the matched-filter score is independent of
    # the PSF's DC level and discretization), but we report results in
    # the units of the *supplied* PSF.  Recover the mean ``mu`` and L2
    # norm ``sigma`` of the original PSF; the score-unit columns from
    # ``_locate_in_chunk`` are rescaled to amplitude units with these
    # at the end.  ``min_contrast`` (an amplitude threshold) becomes a
    # score threshold by multiplying by ``sigma``.
    p_raw = np.asarray(psf, dtype=dtype)
    if p_raw.ndim == 2:
        p_raw = p_raw[np.newaxis, :, :]
    mu = float(p_raw.mean())
    sigma = float(np.linalg.norm(p_raw - mu))
    psf_floor = np.sqrt(np.finfo(p_raw.dtype).eps) * max(float(np.max(np.abs(p_raw - mu))), 1.0)
    if sigma <= psf_floor:
        raise ValueError(
            "`psf` is degenerate (constant or near-constant): its "
            "mean-subtracted L2 norm is ~0, so an amplitude (contrast) "
            "scale is undefined."
        )
    min_contrast_score = min_contrast * sigma

    # Select the channel.  When the video has only one channel, default
    # to it (using the coord value if the channel has a non-default
    # label, e.g. ``"mCherry"``); otherwise the caller must specify which
    # channel to localize.  ``xarray.DataArray.sel`` accepts both a coord
    # label and a positional index, so it handles the "label or integer"
    # dispatch in one step and raises ``KeyError``/``TypeError``/
    # ``IndexError`` for the not-present cases.
    n_channels = int(video.sizes["C"])
    sole_channel = video["C"].values[0] if "C" in video.coords else 0
    if channel is None:
        if n_channels == 1:
            channel = sole_channel
        else:
            raise ValueError(f"Video has {n_channels} channels; please specify `channel`.")
    elif n_channels == 1 and channel != sole_channel:
        raise ValueError(
            f"`channel` is {channel!r} but the video has only one channel ({sole_channel!r})."
        )
    assert channel is not None  # narrow for type checkers; the branches above cover ``None``.
    video = video.sel(C=channel)

    # Restrict to the chosen channel and chunk along the time axis.
    n_frames = int(video.sizes["T"])
    frame_shape = (
        int(video.sizes["Z"]),
        int(video.sizes["Y"]),
        int(video.sizes["X"]),
    )
    chunk_axis = _resolve_chunk_size(chunk_size, n_frames, frame_shape, np.dtype(dtype).itemsize)
    video = video.chunk({"T": chunk_axis})

    # Resolve the per-axis physical origin and step so that the index-space
    # positions emitted by ``_locate_in_chunk`` can be scaled into physical
    # units as ``X = X0 + X_idx * dX``.  The uniform spacing that makes this
    # affine form exact is enforced by ``validate_video``.
    t0, dt = _axis_origin_and_step(video, "T")
    z0, dz = _axis_origin_and_step(video, "Z")
    y0, dy = _axis_origin_and_step(video, "Y")
    x0, dx = _axis_origin_and_step(video, "X")

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
    # the strictly-positive ``min_contrast``), and ``n_active_frames``
    # tells ``_locate_in_chunk`` how many frames are real.
    #
    # ``_locate_in_chunk`` emits index-space positions only; the physical
    # ``t``/``z``/``y``/``x`` columns are computed here from the chunk's
    # coordinate axes and concatenated onto the fit results.
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
        chunk_df = _locate_in_chunk(
            chunk=chunk,
            psf=psf_jax,
            n_active_frames=n_active,
            min_distance=min_distance,
            min_contrast=min_contrast_score,
            sign=sign,
            iterations=iterations,
            atol=atol,
            noise_sigma=noise_sigma,
        )
        results.append(
            _physical_result(
                chunk_df,
                frame_offset=start,
                channel=channel,
                t0=t0,
                dt=dt,
                z0=z0,
                dz=dz,
                y0=y0,
                dy=dy,
                x0=x0,
                dx=dx,
            )
        )
        if on_progress is not None:
            on_progress(end)

    if not results:
        return _empty_result(channel=channel)

    out = polars.concat(results, how="vertical_relaxed")
    # Rescale the score-unit columns from ``_locate_in_chunk`` into the
    # amplitude (data) units of the supplied PSF.  Internally the fit is
    # ``contrast_score * q + bg_fit`` with ``q = (psf - mu) / sigma``;
    # solving for the physical model ``A * psf + B`` gives
    # ``A = contrast_score / sigma`` and ``B = bg_fit - A * mu``.  ``snr``
    # is divided by ``sigma`` so it stays equal to
    # ``contrast / noise_sigma``.
    amp = (polars.col("contrast") / sigma).cast(polars.Float32)
    return out.with_columns(
        amp.alias("contrast"),
        (polars.col("background") - amp * mu).cast(polars.Float32).alias("background"),
        (polars.col("snr") / sigma).cast(polars.Float32).alias("snr"),
    )


def _locate_in_chunk(
    chunk: Float[Array, "B Z Y X"],
    psf: Float[Array, "Z Y X"],
    *,
    n_active_frames: int | None = None,
    min_distance: int = 3,
    min_contrast: float = 1.0,
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
    Px)`` by the caller).  The PSF must also be **normalized**
    (mean-subtracted and L2-normalized, as :func:`_canonicalize_psf`
    does) so that the matched-filter score and the fitted contrast share
    a scale; :func:`locate` guarantees this, but direct callers of
    :func:`_locate_in_chunk` are responsible for normalizing the PSF
    themselves.  Input canonicalization and validation are the
    responsibility of :func:`locate`, not of this function.

    The returned DataFrame carries only index-space coordinates:
    ``chunk_frame`` is the chunk-local frame index (Int32) and ``slice``/
    ``row``/``column`` are subpixel-refined voxel indices (Float32).
    Physical coordinates and the channel label are attached by
    :func:`locate`, which scales the index-space positions into the
    video's coordinate units.

    Parameters
    ----------
    chunk : jax.Array
        A dense ``(B, Z, Y, X)`` array of image data.
    psf : jax.Array
        The 3D point-spread function model, shape ``(Pz, Py, Px)``,
        with ``Pz <= Z``, ``Py <= Y`` and ``Px <= X``.  Must be mean-zero and L2-normalized (see :func:`_canonicalize_psf`)
        so that the matched-filter score and the fitted contrast share
        a scale.
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
        to be reported.  Must be strictly positive; the default is
        ``1.0``.
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
        ``None`` (the default) it is estimated **per frame** from the
        standard deviation of second differences.  The value normalises
        the ``chi2``, ``reduced_chi2`` and ``snr`` output columns.  The
        default is ``None``.

    Returns
    -------
    polars.DataFrame
        A Polars DataFrame with the columns
        ``chunk_frame, slice, row, column, contrast, background,
        snr, chi2, reduced_chi2, n_iter, converged``.  The ``chunk_frame``
        column is Int32; ``slice``/``row``/``column`` are Float32
        (subpixel); the float statistics are Float32; ``n_iter`` is Int32;
        ``converged`` is Boolean.
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
    if Pz > Z:
        raise ValueError(
            f"PSF Z size {Pz} must not exceed data Z size {Z}"
            + ("; use a 2D PSF (or its in-focus plane) for 2D data." if Z == 1 else ".")
        )
    if n_active_frames is None:
        n_active_frames = B
    elif not (0 <= n_active_frames <= B):
        raise ValueError(f"`n_active_frames` must be in [0, {B}], got {n_active_frames}.")
    if noise_sigma is not None and noise_sigma < 0:
        raise ValueError(f"`noise_sigma` must be non-negative, got {noise_sigma!r}.")
    if min_contrast <= 0:
        raise ValueError(f"`min_contrast` must be strictly positive, got {min_contrast!r}.")
    if min_distance < 0:
        raise ValueError(f"`min_distance` must be non-negative, got {min_distance!r}.")
    if iterations < 0:
        raise ValueError(f"`iterations` must be non-negative, got {iterations!r}.")

    # 1+2. Matched-filter score, non-maximum suppression and side-lobe
    #      explain-away produce the boolean (B, Z, Y, X) peak mask for the
    #      whole chunk; one eager ``argwhere`` then yields the integer
    #      coordinates of every detection.  Host syncs are per chunk, not
    #      per frame: one scalar per batch of explain-away rounds (usually
    #      one batch) plus the ``argwhere``.
    is_peak = _peak_mask_batch(
        chunk,
        psf,
        min_distance=min_distance,
        min_contrast=min_contrast,
        sign=sign,
        lobe_tolerance=jnp.asarray(_sidelobe_tolerance(np.asarray(psf)), dtype=chunk.dtype),
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
        return _empty_chunk_result()

    chunk_frame = np.ascontiguousarray(coords_np[:, 0], dtype=np.int32)
    slice_peaks = np.ascontiguousarray(coords_np[:, 1], dtype=np.int32)
    row_peaks = np.ascontiguousarray(coords_np[:, 2], dtype=np.int32)
    column_peaks = np.ascontiguousarray(coords_np[:, 3], dtype=np.int32)

    # 3. Batched stamp extraction: pad the whole chunk once, then read every
    #    stamp with a single advanced-indexing gather (one device dispatch for
    #    all emitters, instead of one eager pad+slice per emitter).
    stamps = _extract_stamps_batch(
        chunk, chunk_frame, slice_peaks, row_peaks, column_peaks, (Pz, Py, Px)
    )

    # Resolve the per-emitter noise standard deviation.  When the caller
    # does not supply one, it is estimated per frame from the standard
    # deviation of second differences (a single-pass O(n) estimator) and
    # gathered per emitter by frame index.  A supplied scalar is broadcast
    # across all emitters.  The result is a (n_emitters,) device array
    # kept resident so no extra host sync is needed before the batched fit.
    if noise_sigma is None:
        per_frame_sigma = _estimate_noise_sigma(chunk, n_active_frames)  # (n_active_frames,)
        noise_sigma_arr = per_frame_sigma[jnp.asarray(chunk_frame)]  # (n_emitters,)
    else:
        noise_sigma_arr = jnp.full((n_emitters,), noise_sigma, dtype=chunk.dtype)

    # 4. Batched Levenberg--Marquardt refinement.  Stamps are processed in
    #    sub-batches whose size is the next power of two (capped at
    #    ``_MAX_EMITTERS_PER_BATCH``) so that the JIT-compiled fitting kernel
    #    specialises on a small set of shapes rather than one per distinct
    #    emitter count.  All sub-batch dispatches are queued asynchronously;
    #    sub-batch results are concatenated before reading them back to
    #    reduce the number of host syncs.
    batch_size = 1 << (n_emitters - 1).bit_length()  # next power of two
    batch_size = min(batch_size, _MAX_EMITTERS_PER_BATCH)
    fit_keys = [
        "z_offset",
        "y_offset",
        "x_offset",
        "contrast",
        "background",
        "chi2",
        "reduced_chi2",
        "snr",
    ]
    fit_stacks: list[jax.Array] = []
    converged_stacks: list[jax.Array] = []
    n_iter_stacks: list[jax.Array] = []
    for i in range(0, n_emitters, batch_size):
        sub_stamps = stamps[i : i + batch_size]
        sub_sigma = noise_sigma_arr[i : i + batch_size]
        n_sub = sub_stamps.shape[0]
        if n_sub < batch_size:
            sub_stamps = jnp.pad(
                sub_stamps,
                ((0, batch_size - n_sub), (0, 0), (0, 0), (0, 0)),
                mode="constant",
            )
            sub_sigma = jnp.pad(sub_sigma, (0, batch_size - n_sub), mode="constant")
        sub_fit = _fit_emitters_batch(
            sub_stamps,
            psf,
            iterations=iterations,
            atol=atol,
            noise_sigma=sub_sigma,
        )
        sub_stack = jnp.stack([sub_fit[k] for k in fit_keys])  # (8, batch)
        fit_stacks.append(sub_stack[:, :n_sub])
        converged_stacks.append(sub_fit["converged"][:n_sub])
        n_iter_stacks.append(sub_fit["n_iter"][:n_sub])

    # Concatenate sub-batch results before reading them back to reduce
    # the number of host syncs.
    fit_np = np.asarray(jnp.concatenate(fit_stacks, axis=1), dtype=np.float32)
    z_offsets, y_offsets, x_offsets = fit_np[0], fit_np[1], fit_np[2]
    contrast_col = np.ascontiguousarray(fit_np[3])
    background_col = np.ascontiguousarray(fit_np[4])
    chi2_col = np.ascontiguousarray(fit_np[5])
    reduced_chi2_col = np.ascontiguousarray(fit_np[6])
    snr_col = np.ascontiguousarray(fit_np[7])
    n_iter_col = np.asarray(jnp.concatenate(n_iter_stacks), dtype=np.int32)
    converged_col = np.asarray(jnp.concatenate(converged_stacks), dtype=bool)

    # 5. Compose the index-space output table.  Subpixel-refined voxel
    #    indices are the integer peak pixel plus the LM subpixel offset;
    #    they stay Float32 (the offset's precision).  Physical coordinates
    #    are derived by ``locate`` from the video's coordinate axes.
    return polars.DataFrame(
        {
            "chunk_frame": chunk_frame,
            "slice": (slice_peaks.astype(np.float32) + z_offsets).astype(np.float32),
            "row": (row_peaks.astype(np.float32) + y_offsets).astype(np.float32),
            "column": (column_peaks.astype(np.float32) + x_offsets).astype(np.float32),
            "contrast": contrast_col,
            "background": background_col,
            "snr": snr_col,
            "chi2": chi2_col,
            "reduced_chi2": reduced_chi2_col,
            "n_iter": n_iter_col,
            "converged": converged_col,
        },
        schema={
            "chunk_frame": polars.Int32,
            "slice": polars.Float32,
            "row": polars.Float32,
            "column": polars.Float32,
            "contrast": polars.Float32,
            "background": polars.Float32,
            "snr": polars.Float32,
            "chi2": polars.Float32,
            "reduced_chi2": polars.Float32,
            "n_iter": polars.Int32,
            "converged": polars.Boolean,
        },
    )


# ---------------------------------------------------------------------------
# Matched filter, non-maximum suppression and side-lobe explain-away
# ---------------------------------------------------------------------------


def _peak_mask_batch(
    chunk: Float[Array, "B Z Y X"],
    psf: Float[Array, "Pz Py Px"],
    *,
    min_distance: int,
    min_contrast: float,
    sign: Literal["both", "positive", "negative"],
    lobe_tolerance: Float[Array, "2 Qz Qy Qx"],
) -> Bool[Array, "B Z Y X"]:
    """
    Compute the matched-filter score and the boolean peak mask for every
    frame in ``chunk``.

    The detection proceeds in four steps:

    1. **Matched filter.**  Each frame is edge-padded by the PSF
       half-extent and correlated with the PSF in ``"valid"`` mode, so
       the score at ``(z, y, x)`` is exactly the inner product of the PSF
       with the (edge-padded) stamp that :func:`_extract_stamps_batch`
       extracts there.  Edge padding -- rather than the implicit zero
       padding of ``mode="same"`` -- keeps a constant background
       cancelling against the mean-zero PSF all the way to the frame
       border; zero padding would turn any non-zero background into
       spurious border detections.
    2. **Non-maximum suppression.**  A pixel is a candidate if it is the
       extremum of its ``(2*min_distance+1)**3`` spatial window (extent
       ``1`` along the frame axis, so detections never bleed across
       frames) and its magnitude exceeds ``min_contrast``.
    3. **Tie breaking.**  Exactly tied extrema within one window (e.g. an
       emitter centred between two pixels) would otherwise all pass the
       ``score == local_extremum`` test; only the first in raster order
       is kept.
    4. **Side-lobe explain-away.**  The score of an emitter is
       (approximately) its peak score times the normalized PSF
       autocorrelation, which has side lobes: always of opposite sign
       (the PSF is mean-zero), and also of the same sign for ringing
       PSFs (e.g. iSCAT).  Candidates are resolved greedily, so that
       only *accepted* emitters explain away the lobes of other
       candidates -- a candidate never explains away a stronger one.
       See :func:`_explain_away_rounds`.  The step is applied regardless
       of ``sign``, because a dark object's positive side lobes must not
       be reported as positive emitters either.

    Steps 1--3 run in one JIT kernel.  Step 4 runs in fixed-size batches
    of greedy rounds; the host only launches another batch while
    candidates remain unresolved (rarely more than one batch is needed).
    """
    scores, cand_pos, cand_neg = _peak_candidates_batch(
        chunk, psf, min_distance=min_distance, min_contrast=min_contrast
    )
    unresolved = cand_pos | cand_neg
    amplitude = jnp.zeros_like(scores)
    n_unresolved = None
    while True:
        unresolved, amplitude, count = _explain_away_rounds(
            scores,
            cand_pos,
            cand_neg,
            unresolved,
            amplitude,
            psf,
            lobe_tolerance,
            min_contrast=min_contrast,
        )
        # Every round resolves at least the leading candidate of each
        # frame, so the count only stalls for non-finite scores; the
        # remaining candidates are then rejected.
        previous, n_unresolved = n_unresolved, int(count)
        if n_unresolved == 0 or n_unresolved == previous:
            break

    if sign == "positive":
        return cand_pos & (amplitude > 0)
    if sign == "negative":
        return cand_neg & (amplitude < 0)
    return (cand_pos & (amplitude > 0)) | (cand_neg & (amplitude < 0))


@jax.jit(static_argnames=["min_distance"])
def _peak_candidates_batch(
    chunk: Float[Array, "B Z Y X"],
    psf: Float[Array, "Pz Py Px"],
    *,
    min_distance: int,
    min_contrast: float,
) -> tuple[Float[Array, "B Z Y X"], Bool[Array, "B Z Y X"], Bool[Array, "B Z Y X"]]:
    """
    Matched-filter scores and positive/negative candidate masks (steps
    1--3 of :func:`_peak_mask_batch`).
    """
    Pz, Py, Px = psf.shape
    hz, hy, hx = Pz // 2, Py // 2, Px // 2

    # 1. Matched filter on the edge-padded frames.
    padded = jnp.pad(
        chunk,
        ((0, 0), (hz, Pz - hz - 1), (hy, Py - hy - 1), (hx, Px - hx - 1)),
        mode="edge",
    )
    scores = jax.vmap(lambda frame: jsignal.correlate(frame, psf, mode="valid", method="fft"))(
        padded
    )  # (B, Z, Y, X)

    # 2. Non-maximum suppression (separable box filters).
    nms_window = (1, 2 * min_distance + 1, 2 * min_distance + 1, 2 * min_distance + 1)
    neg_inf = jnp.asarray(-jnp.inf, dtype=scores.dtype)
    pos_inf = jnp.asarray(jnp.inf, dtype=scores.dtype)
    local_max = _box_reduce(scores, neg_inf, jax.lax.max, nms_window)
    local_min = _box_reduce(scores, pos_inf, jax.lax.min, nms_window)
    cand_pos = (scores == local_max) & (scores > min_contrast)
    cand_neg = (scores == local_min) & (scores < -min_contrast)

    # 3. Deterministic tie breaking.
    cand_pos = _break_ties(cand_pos, nms_window)
    cand_neg = _break_ties(cand_neg, nms_window)
    return scores, cand_pos, cand_neg


@jax.jit(static_argnames=["n_rounds"])
def _explain_away_rounds(
    scores: Float[Array, "B Z Y X"],
    cand_pos: Bool[Array, "B Z Y X"],
    cand_neg: Bool[Array, "B Z Y X"],
    unresolved: Bool[Array, "B Z Y X"],
    amplitude: Float[Array, "B Z Y X"],
    psf: Float[Array, "Pz Py Px"],
    lobe_tolerance: Float[Array, "2 Qz Qy Qx"],
    *,
    min_contrast: float,
    n_rounds: int = 2,
) -> tuple[Bool[Array, "B Z Y X"], Float[Array, "B Z Y X"], Int[Array, ""]]:
    """
    Run ``n_rounds`` rounds of the greedy side-lobe explain-away.

    ``amplitude`` holds the signed amplitude of every accepted emitter
    (zero elsewhere); ``unresolved`` marks the candidates not yet
    accepted or rejected.  A candidate's *remainder* is its score minus
    the lobes predicted from all accepted emitters (their amplitudes
    convolved with the normalized PSF autocorrelation).  Its *excess* is
    by how much the remainder, with the candidate's sign, exceeds
    ``min_contrast`` plus a tolerance: the accepted opposite-sign and
    same-sign amplitudes convolved with the per-offset profiles
    ``lobe_tolerance[0]`` and ``lobe_tolerance[1]``, which bound the
    mismatch between the on-grid prediction and real emitters (see
    :func:`_sidelobe_tolerance`).

    The greedy resolves candidates in order of decreasing excess and is
    parallelized: in every round, each unresolved candidate whose excess
    is the largest among the unresolved candidates within the
    autocorrelation's footprint is resolved -- all such candidates are
    too far apart to affect each other.  It is accepted, with its
    remainder as amplitude, if its excess is positive, and rejected
    otherwise; rejected candidates explain nothing away.  In addition,
    candidates are rejected right away if no unresolved candidate within
    their footprint has a positive excess (see below).

    A simultaneous (Jacobi) update, in which every candidate's lobes are
    subtracted from every other candidate, does not work for PSFs with
    strong side lobes: the lobe candidates around a bright emitter then
    jointly explain away most of the emitter itself, which in turn
    leaves its lobes unexplained.

    Returns the updated ``unresolved`` mask and ``amplitude``, and the
    largest number of unresolved candidates in any frame.
    """
    acf = jsignal.correlate(psf, psf, mode="full", method="fft")
    acf = acf / jnp.max(acf)  # (2Pz-1, 2Py-1, 2Px-1), centre value 1
    tol_opposite, tol_same = lobe_tolerance[0], lobe_tolerance[1]
    zero = jnp.asarray(0, dtype=scores.dtype)
    neg_inf = jnp.asarray(-jnp.inf, dtype=scores.dtype)
    # Two candidates interact iff their offset lies within the acf (and
    # tolerance) support, so this window is the greedy's footprint.
    Qz, Qy, Qx = acf.shape
    footprint = (1, Qz, Qy, Qx)

    # A positive candidate's tolerance is ``mag_neg * tol_opposite +
    # mag_pos * tol_same`` (convolved).  With ``|a| = mag_pos + mag_neg``
    # and ``a = mag_pos - mag_neg``, this is ``|a| * tol_even + a *
    # tol_odd``, and a negative candidate's is ``|a| * tol_even - a *
    # tol_odd``.  The lobes, ``a * acf``, share the spectrum of ``a``, so
    # a round costs two forward and three inverse FFTs.
    axes = (1, 2, 3)
    fft_shape = tuple(n + k - 1 for n, k in zip(scores.shape[1:], acf.shape, strict=True))
    same = (slice(None),) + tuple(
        slice((k - 1) // 2, (k - 1) // 2 + n)
        for n, k in zip(scores.shape[1:], acf.shape, strict=True)
    )
    acf_hat, tol_odd_hat, tol_even_hat = jnp.fft.rfftn(
        jnp.stack([acf, (tol_same - tol_opposite) / 2, (tol_same + tol_opposite) / 2]),
        s=fft_shape,
        axes=axes,
    )

    def spread(spectrum: jax.Array) -> jax.Array:
        return jnp.fft.irfftn(spectrum, s=fft_shape, axes=axes)[same].astype(scores.dtype)

    for _ in range(n_rounds):
        amp_hat = jnp.fft.rfftn(amplitude, s=fft_shape, axes=axes)
        abs_hat = jnp.fft.rfftn(jnp.abs(amplitude), s=fft_shape, axes=axes)
        # Only accepted emitters carry an amplitude, and an unresolved
        # candidate has none, so this never subtracts its own peak.
        remainder = scores - spread(amp_hat * acf_hat)
        tol_even = spread(abs_hat * tol_even_hat)
        tol_odd = spread(amp_hat * tol_odd_hat)
        excess = (
            jnp.where(cand_pos, remainder - tol_odd, -remainder + tol_odd) - tol_even - min_contrast
        )
        key = jnp.where(unresolved, excess, neg_inf)
        local_max = _box_reduce(key, neg_inf, jax.lax.max, footprint)
        dominant = _break_ties(unresolved & (key == local_max), footprint)
        accept = dominant & (key > zero)
        # A key only changes when a candidate within the footprint is
        # accepted.  If no unresolved candidate in the footprint can
        # currently be accepted (all keys <= 0), this one is rejected
        # right away instead of waiting for its turn, which would take
        # one round per candidate.  This deviates from the strict greedy
        # order only if accepting a candidate *outside* the footprint
        # first lifts one inside it above the threshold, and then only by
        # rejecting more.
        hopeless = unresolved & (local_max <= zero)
        amplitude = jnp.where(accept, remainder, amplitude)
        unresolved = unresolved & ~dominant & ~hopeless

    return unresolved, amplitude, jnp.max(jnp.sum(unresolved, axis=(1, 2, 3)))


def _box_reduce(
    x: jax.Array,
    init: jax.Array,
    op: Callable[[jax.Array, jax.Array], jax.Array],
    window: tuple[int, ...],
) -> jax.Array:
    """
    Apply a ``"same"``-padded box reduction (max or min) over ``window``.

    Max and min over a box are separable, so the reduction is applied one
    axis at a time: the cost is ``O(sum(window))`` per element instead of
    ``O(prod(window))``.  Every window extent must be odd.
    """
    ones = (1,) * x.ndim
    for axis, size in enumerate(window):
        if size > 1:
            dims = tuple(size if a == axis else 1 for a in range(x.ndim))
            x = jax.lax.reduce_window(x, init, op, dims, ones, "same")
    return x


def _break_ties(
    mask: Bool[Array, "B Z Y X"],
    window: tuple[int, int, int, int],
) -> Bool[Array, "B Z Y X"]:
    """
    Keep only the first (raster-order) peak among peaks sharing a window.

    Two pixels that both equal the extremum of their (symmetric) NMS
    window and lie within each other's window necessarily have exactly
    equal scores, so this only affects ties.  Each peak is dropped if any
    other peak within ``window`` has a smaller per-frame linear index.
    """
    _, Z, Y, X = mask.shape
    index = jnp.arange(Z * Y * X, dtype=jnp.int32).reshape(1, Z, Y, X)
    sentinel = jnp.asarray(jnp.iinfo(jnp.int32).max, dtype=jnp.int32)
    keyed = jnp.where(mask, index, sentinel)
    first = _box_reduce(keyed, sentinel, jax.lax.min, window)
    return mask & (first == index)


@functools.lru_cache(maxsize=16)
def _sidelobe_tolerance_cached(
    psf_bytes: bytes, shape: tuple[int, int, int], dtype: str
) -> npt.NDArray[np.floating]:
    psf = np.frombuffer(psf_bytes, dtype=dtype).reshape(shape)
    P = np.array(shape)
    profiles = np.zeros((2, *(2 * P - 1)), dtype=np.float64)  # (opposite, same)
    q = psf.astype(np.float64) - psf.mean(dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if norm == 0.0:
        return profiles.astype(psf.dtype)
    q /= norm
    acf = scipy.signal.correlate(q, q, mode="full", method="fft")
    acf /= acf.max()

    # Synthetic emitter: the PSF, padded with its mean border value (a PSF
    # that has decayed to a flat background outside its window).
    border = np.ones(shape, dtype=bool)
    interior = tuple(
        slice(None) if size == 1 else slice(1, -1) if size > 2 else slice(0, 0) for size in shape
    )
    border[interior] = False
    background = float(q[border].mean())
    padded = np.pad(q, [(p, p) for p in shape], mode="constant", constant_values=background)
    # Shifting the emitter and correlating it with ``q`` is a single
    # product in Fourier space.  On the ``3P`` canvas, the circular
    # correlation at lags ``0..2P`` equals the ``"valid"`` correlation.
    spectrum = np.fft.fftn(padded) * np.conj(np.fft.fftn(q, s=padded.shape, axes=(0, 1, 2)))
    k = np.meshgrid(*[np.fft.fftfreq(s) for s in padded.shape], indexing="ij")
    steps = [(-0.5, -0.25, 0.0, 0.25, 0.5) if size > 1 else (0.0,) for size in shape]
    phases = [
        np.exp(-2j * np.pi * sum(ki * si for ki, si in zip(k, shift, strict=True)))
        for shift in itertools.product(*steps)
    ]

    for phase in phases:
        score = np.fft.ifftn(spectrum * phase).real[tuple(slice(0, 2 * p + 1) for p in shape)]
        peak = np.array(np.unravel_index(np.argmax(score), score.shape))
        peak_score = float(score[tuple(peak)])
        if peak_score <= 0.0:
            continue
        # Compare the score with ``peak_score * acf`` centred on the
        # peak (acf centre: P - 1) over their overlap.
        lo = np.maximum(peak - (P - 1), 0)
        hi = np.minimum(peak + P, score.shape)
        score_part = score[tuple(slice(a, b) for a, b in zip(lo, hi, strict=True))]
        acf_part = tuple(
            slice(a - c + p - 1, b - c + p - 1) for a, b, c, p in zip(lo, hi, peak, P, strict=True)
        )
        mismatch = (score_part - peak_score * acf[acf_part]) / peak_score
        excess = np.stack([np.maximum(-mismatch, 0.0), np.maximum(mismatch, 0.0)])
        profiles[(slice(None), *acf_part)] = np.maximum(profiles[(slice(None), *acf_part)], excess)
    # The candidate itself (offset 0) never needs a same-sign tolerance.
    profiles[(1, *(P - 1))] = 0.0
    return (_SIDELOBE_MARGIN * profiles).astype(psf.dtype)


def _sidelobe_tolerance(psf: npt.ArrayLike) -> npt.NDArray[np.floating]:
    """
    Per-offset tolerance profiles of the side-lobe explain-away step.

    :func:`_peak_mask_batch` predicts the side lobes of an emitter as its
    peak score times the on-grid PSF autocorrelation.  Real emitters
    deviate from that prediction, because they sit between pixels and
    because the PSF continues beyond its window.  This function
    simulates emitters at subpixel offsets of ``0``, ``+-1/4`` and
    ``+-1/2`` voxel along every axis the PSF extends over.  For every
    offset ``d`` from the detected peak it records the largest amount,
    relative to the peak score, by which the true score exceeds the
    prediction

    - in the direction *opposite* to the emitter's sign (profile ``0``,
      relevant for opposite-sign candidates), and
    - in the direction of the emitter's sign (profile ``1``, relevant for
      same-sign candidates; zero at ``d = 0``).

    The result has shape ``(2, 2*Pz-1, 2*Py-1, 2*Px-1)``, is multiplied
    by the safety factor :data:`_SIDELOBE_MARGIN`, depends only on the
    PSF, and is cached.
    """
    arr = np.ascontiguousarray(np.asarray(psf))
    return _sidelobe_tolerance_cached(arr.tobytes(), tuple(arr.shape), arr.dtype.str)


# ---------------------------------------------------------------------------
# Stamp extraction (batched)
# ---------------------------------------------------------------------------


def _extract_stamps_batch(
    chunk: Float[Array, "B Z Y X"],
    chunk_frame: npt.NDArray[np.int32],
    slice_peaks: npt.NDArray[np.int32],
    row_peaks: npt.NDArray[np.int32],
    column_peaks: npt.NDArray[np.int32],
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
    n = chunk_frame.shape[0]

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
    b_g = np.broadcast_to(chunk_frame[:, None, None, None], (n, Pz, Py, Px))
    z_g = slice_peaks[:, None, None, None] + oz[None]
    y_g = row_peaks[:, None, None, None] + oy[None]
    x_g = column_peaks[:, None, None, None] + ox[None]
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

    ``noise_sigma`` is the per-emitter noise standard deviation (a
    ``(batch,)`` array, one value per emitter) used to normalise each
    emitter's ``chi2``, ``reduced_chi2`` and ``snr`` outputs.  It is
    mapped alongside the stamps (``in_axes=(0, 0)``).

    Returns a dictionary of per-emitter outputs:
    ``contrast``, ``background``, ``z_offset``, ``y_offset``,
    ``x_offset``, ``converged``, ``n_iter``, ``chi2``,
    ``reduced_chi2``, ``snr``.
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
        in_axes=(0, 0),
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
        return full[None, pad : pad + Py, pad : pad + Px]

    return shifted_psf, 2


def _lm_refine(
    params0: jax.Array,
    residual: Callable[[jax.Array], jax.Array],
    cost: Callable[[jax.Array], jax.Array],
    *,
    lower: jax.Array,
    upper: jax.Array,
    iterations: int,
    atol: float,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    Run a fixed-length damped Gauss-Newton (Levenberg--Marquardt) scan.

    Each step solves the Gauss--Newton normal equations
    ``(J^T J + lambda * diag(J^T J)) delta = -J^T r`` with
    ``J = jacfwd(residual)``; the candidate ``params + delta`` is
    projected onto the box ``[lower, upper]`` before its cost is
    evaluated, and the damping ``lambda`` is halved on accept and
    doubled on reject.  Once an emitter's
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
        candidate_params = jnp.clip(params + delta, lower, upper)
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

    # Initial guesses: contrast from the stamp centre minus the
    # background, background from the stamp median, and zero subpixel
    # shifts.
    bg0 = jnp.median(stamp)
    amp0 = stamp[Pz // 2, Py // 2, Px // 2] - bg0
    shift0 = jnp.zeros(n_shift, dtype=stamp.dtype)
    params0 = jnp.concatenate([jnp.array([amp0, bg0], dtype=stamp.dtype), shift0])

    # Contrast and background are unbounded; the subpixel shifts are
    # confined to ``[-_MAX_SUBPIXEL_SHIFT, _MAX_SUBPIXEL_SHIFT]`` so the
    # fit cannot drift away from the detected peak.
    inf = jnp.full(2, jnp.inf, dtype=stamp.dtype)
    bound = jnp.full(n_shift, _MAX_SUBPIXEL_SHIFT, dtype=stamp.dtype)
    upper = jnp.concatenate([inf, bound])
    lower = -upper

    final_params, final_cost, converged, n_iter = _lm_refine(
        params0,
        residual,
        cost,
        lower=lower,
        upper=upper,
        iterations=iterations,
        atol=atol,
    )
    # A fit that ends on the shift bound wanted to move further away from
    # the detected peak; its result is not trustworthy.
    converged = converged & jnp.all(jnp.abs(final_params[2:]) < _MAX_SUBPIXEL_SHIFT)
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
    Coerce a PSF model to the canonical normalized 3D ``(Pz, Py, Px)`` form.

    A 2D ``(Py, Px)`` PSF (widefield) is promoted to ``(1, Py, Px)``.
    Any other rank is rejected.  The PSF is then **normalized** so that
    the matched-filter score and the fitted contrast land on the same
    scale: the mean is subtracted (making the PSF mean-zero, so a flat
    background contributes nothing to the correlation score) and the
    result is divided by its L2 norm (so a unit-contrast emitter
    produces a unit peak score).  The result is cast to ``dtype`` when
    supplied, otherwise the input dtype is preserved.

    Normalization matters because :func:`locate` uses the PSF both as
    the matched-filter kernel (for detection) and as the fitting
    template (for refinement).  With a mean-zero, L2-normalized PSF the
    peak matched-filter score of a unit-contrast emitter is exactly 1,
    so ``min_contrast`` becomes a direct threshold on the fitted
    ``contrast`` column rather than on a PSF- and background-dependent
    score.  A constant or all-zero PSF (L2 norm zero) is left unchanged
    after mean subtraction, with a precision-relative floor guarding the
    division.

    Parameters
    ----------
    psf : array-like
        The PSF model, 2D or 3D.
    dtype : numpy dtype, optional
        The dtype to cast the PSF to.

    Returns
    -------
    numpy.ndarray
        A normalized 3D ``(Pz, Py, Px)`` array of the requested dtype.

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
    # Mean-subtract (kills the background contribution to the matched-filter
    # score) and L2-normalize (so a unit-contrast emitter scores 1.0).  The
    # floor on the norm avoids a divide-by-zero for a constant PSF.
    arr = arr - arr.mean()
    norm = np.linalg.norm(arr)
    floor = np.sqrt(np.finfo(arr.dtype).eps) * max(float(np.max(np.abs(arr))), 1.0)
    arr = arr / max(norm, floor)
    return arr


def _estimate_noise_sigma(
    chunk: Float[Array, "B Z Y X"],
    n_active_frames: int,
) -> jax.Array:
    """
    Estimate the per-pixel noise standard deviation of ``chunk`` per frame.

    Returns a ``(n_active_frames,)`` array with one sigma value per
    frame.

    A second-difference filter (taps ``[1, -2, 1]``) cancels smooth
    signal and constant background, so its output is dominated by
    noise.  Second differences are computed along every spatial axis
    with at least three samples, pooled per frame, and the per-frame
    standard deviation is divided by the filter's L2 gain
    ``sqrt(sum(taps**2))`` (``sqrt(6)`` for ``[1, -2, 1]``) to recover
    the underlying noise sigma.

    This is a single-pass O(n) estimator (one fused reduction per
    frame, no sort).  It is non-robust to bright emitter pixels, but
    emitters are sparse and this is a fallback for when the caller does
    not supply a known ``noise_sigma``; a supplied value always takes
    precedence.  On pure Gaussian noise the standard deviation is a
    lower-variance estimator than the median absolute deviation.

    A tiny floor guards against noise-free data, where the estimate is
    exactly zero; the per-emitter precision floor in :func:`_fit_outputs`
    dominates in practice.
    """
    active = chunk[:n_active_frames]  # (B, Z, Y, X)
    B = active.shape[0]
    Z, Y, X = active.shape[1], active.shape[2], active.shape[3]
    taps = jnp.array([1.0, -2.0, 1.0], dtype=active.dtype)
    hf_gain = jnp.sqrt(jnp.sum(taps**2))  # sqrt(6)
    diffs: list[jax.Array] = []
    if X >= 3:
        diffs.append((active[..., 2:] - 2 * active[..., 1:-1] + active[..., :-2]).reshape(B, -1))
    if Y >= 3:
        diffs.append(
            (active[:, :, 2:, :] - 2 * active[:, :, 1:-1, :] + active[:, :, :-2, :]).reshape(B, -1)
        )
    if Z >= 3:
        diffs.append(
            (active[:, 2:, :, :] - 2 * active[:, 1:-1, :, :] + active[:, :-2, :, :]).reshape(B, -1)
        )
    if not diffs:
        return jnp.full((B,), jnp.finfo(active.dtype).tiny, dtype=active.dtype)
    d = jnp.concatenate(diffs, axis=1)  # (B, n_samples)
    sigma = jnp.std(d, axis=1) / hf_gain  # (B,)
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


def _empty_result(
    channel: int | str | float = 0,
) -> polars.DataFrame:
    """
    Return an empty Polars DataFrame with the ``locate`` output schema.

    The ``channel`` column is typed to match ``channel`` (via
    :func:`_channel_scalar_and_dtype`); every other column follows the
    fixed schema below.
    """
    c_scalar, c_dtype = _channel_scalar_and_dtype(channel)
    schema = {
        "t": polars.Float64,
        "channel": c_dtype,
        "z": polars.Float64,
        "y": polars.Float64,
        "x": polars.Float64,
        "frame": polars.Int32,
        "slice": polars.Float32,
        "row": polars.Float32,
        "column": polars.Float32,
        "contrast": polars.Float32,
        "background": polars.Float32,
        "snr": polars.Float32,
        "chi2": polars.Float32,
        "reduced_chi2": polars.Float32,
        "n_iter": polars.Int32,
        "converged": polars.Boolean,
    }
    return polars.DataFrame(schema=schema).with_columns(
        polars.lit(c_scalar).cast(c_dtype).alias("channel"),
    )


def _empty_chunk_result() -> polars.DataFrame:
    """Return an empty Polars DataFrame with the ``_locate_in_chunk`` schema."""
    return polars.DataFrame(
        schema={
            "chunk_frame": polars.Int32,
            "slice": polars.Float32,
            "row": polars.Float32,
            "column": polars.Float32,
            "contrast": polars.Float32,
            "background": polars.Float32,
            "snr": polars.Float32,
            "chi2": polars.Float32,
            "reduced_chi2": polars.Float32,
            "n_iter": polars.Int32,
            "converged": polars.Boolean,
        }
    )


def _physical_result(
    chunk_df: polars.DataFrame,
    *,
    frame_offset: int,
    channel: int | str | float,
    t0: float,
    dt: float,
    z0: float,
    dz: float,
    y0: float,
    dy: float,
    x0: float,
    dx: float,
) -> polars.DataFrame:
    """
    Scale a chunk's index-space detections into the public physical schema.

    ``chunk_df`` is the DataFrame returned by :func:`_locate_in_chunk`
    (columns ``chunk_frame``, ``slice``, ``row``, ``column`` and the fit
    statistics).  Physical coordinates are computed in float64 as
    ``X = X0 + X_idx * dX`` using the video's per-axis origin and step;
    the global ``frame`` is the chunk-local ``chunk_frame`` plus the
    chunk's frame offset.  The ``channel`` label column is attached as a
    constant.
    """
    n = chunk_df.height
    if n == 0:
        return _empty_result(channel)

    c_scalar, c_dtype = _channel_scalar_and_dtype(channel)
    chunk_frame = chunk_df["chunk_frame"].to_numpy()
    frame = (chunk_frame + np.int32(frame_offset)).astype(np.int32)
    slice_arr = chunk_df["slice"].to_numpy()
    row_arr = chunk_df["row"].to_numpy()
    column_arr = chunk_df["column"].to_numpy()
    t = np.float64(t0) + frame.astype(np.float64) * np.float64(dt)
    z = np.float64(z0) + slice_arr.astype(np.float64) * np.float64(dz)
    y = np.float64(y0) + row_arr.astype(np.float64) * np.float64(dy)
    x = np.float64(x0) + column_arr.astype(np.float64) * np.float64(dx)

    return chunk_df.select(
        polars.Series("t", t, dtype=polars.Float64),
        polars.Series("channel", np.full(n, c_scalar), dtype=c_dtype),
        polars.Series("z", z, dtype=polars.Float64),
        polars.Series("y", y, dtype=polars.Float64),
        polars.Series("x", x, dtype=polars.Float64),
        polars.Series("frame", frame, dtype=polars.Int32),
        polars.Series("slice", slice_arr, dtype=polars.Float32),
        polars.Series("row", row_arr, dtype=polars.Float32),
        polars.Series("column", column_arr, dtype=polars.Float32),
        polars.col("contrast"),
        polars.col("background"),
        polars.col("snr"),
        polars.col("chi2"),
        polars.col("reduced_chi2"),
        polars.col("n_iter"),
        polars.col("converged"),
    )


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
    return max(1, min(int(chunk_size), n_frames))
