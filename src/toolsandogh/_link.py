"""
Link per-frame emitter localizations into trajectories.

The :func:`link` function consumes the per-frame emitter table produced by
:func:`toolsandogh.locate` and returns the same table augmented with a
``particle_id`` column.  Trajectories are formed frame-by-frame: for every
detected frame, the still-open trajectories are matched to the new
detections with the Hungarian algorithm
(``scipy.optimize.linear_sum_assignment``), and unmatched detections
start new trajectories.

Empty frames (frames that contain no detections) are not present in the
input.  They are inferred from the gap between consecutive detected
frames and count against ``memory`` for every active trajectory.

Positions are compared in either physical (micrometre) or index (pixel)
space, as selected by the search-range argument; frames are always
counted by the integer ``t_idx`` column so that gaps and ``memory`` are
independent of the frame rate.
"""

from collections.abc import Sequence

import numpy as np
import polars
import scipy.optimize

# Spatial column names, in the canonical ``(z, y, x)`` order, for the two
# coordinate spaces ``link`` can operate in.
_SPATIAL_COLS_PHYS = ("z", "y", "x")
_SPATIAL_COLS_PX = ("z_idx", "y_idx", "x_idx")


def _parse_search_range(
    search_range: float | int | Sequence[float | int],
    *,
    param_name: str,
) -> np.ndarray:
    """
    Turn a scalar or 3-tuple search range into a ``(3,)`` float64 array.

    A scalar is broadcast to every axis.  A sequence of length 3 is
    interpreted as ``(z, y, x)``.  Any other length or type is rejected.
    All entries must be positive and finite.
    """
    if isinstance(search_range, (int, float, np.integer, np.floating)):
        val = float(search_range)
        if not np.isfinite(val) or val <= 0:
            raise ValueError(f"`{param_name}` must be positive and finite, got {search_range!r}.")
        return np.full(3, val, dtype=np.float64)

    try:
        seq = list(search_range)
    except TypeError as exc:
        raise ValueError(
            f"`{param_name}` must be a scalar or a 3-tuple of scalars, "
            f"got {type(search_range).__name__}."
        ) from exc

    if len(seq) != 3:
        raise ValueError(
            f"`{param_name}` must be a scalar or a 3-tuple ``(z, y, x)``, got length {len(seq)}."
        )
    out = np.asarray(seq, dtype=np.float64)
    if not np.all(np.isfinite(out)) or np.any(out <= 0):
        raise ValueError(f"`{param_name}` entries must all be positive and finite, got {seq!r}.")
    return out


def _resolve_search_ranges(
    search_range_pixels: float | int | Sequence[float | int] | None,
    search_range_micrometers: float | int | Sequence[float | int] | None,
    locs: polars.DataFrame,
) -> tuple[np.ndarray, tuple[str, str, str]]:
    """
    Resolve the two mutually-exclusive search-range arguments.

    Exactly one of ``search_range_pixels`` and ``search_range_micrometers``
    must be supplied.  The chosen mode determines which spatial columns
    are required and read:

    * micrometre mode reads the physical ``z``/``y``/``x`` columns;
    * pixel mode reads the index-space ``z_idx``/``y_idx``/``x_idx`` columns.

    Returns ``(ranges, columns)`` where ``ranges`` is a float64 ``(3,)``
    array of per-axis thresholds and ``columns`` is the tuple of column
    names the caller should read for positions.
    """
    if search_range_pixels is None and search_range_micrometers is None:
        raise ValueError(
            "Exactly one of `search_range_pixels` and `search_range_micrometers` must be supplied."
        )
    if search_range_pixels is not None and search_range_micrometers is not None:
        raise ValueError(
            "`search_range_pixels` and `search_range_micrometers` are "
            "mutually exclusive; supply only one."
        )

    if search_range_micrometers is not None:
        cols = _SPATIAL_COLS_PHYS
        param_name = "search_range_micrometers"
        value: float | int | Sequence[float | int] = search_range_micrometers
    else:
        assert search_range_pixels is not None
        cols = _SPATIAL_COLS_PX
        param_name = "search_range_pixels"
        value = search_range_pixels

    missing = set(cols) - set(locs.columns)
    if missing:
        raise ValueError(
            f"`{param_name}` requires the columns {sorted(cols)}, missing: {sorted(missing)}."
        )
    return _parse_search_range(value, param_name=param_name), cols


def link(
    locs: polars.DataFrame,
    *,
    search_range_pixels: float | int | Sequence[float | int] | None = None,
    search_range_micrometers: float | int | Sequence[float | int] | None = None,
    memory: int = 0,
    adaptive_step: float | None = None,
) -> polars.DataFrame:
    """
    Link per-frame emitter localizations into trajectories.

    For every detected frame, the still-open trajectories are matched
    to the new detections with the Hungarian algorithm.  Each detection
    is either appended to an existing trajectory (if it is within the
    search range of the trajectory's last known position) or starts a
    new one.  A trajectory that goes unmatched for more than
    ``memory`` frames is closed.

    Each channel (``c`` column) is linked independently.  Frames are
    counted by the integer ``t_idx`` column, so ``memory`` and any
    adaptive scaling are independent of the frame rate.

    Parameters
    ----------
    locs : polars.DataFrame
        Per-frame emitter table from :func:`toolsandogh.locate`.  Must
        contain the columns ``c`` and ``t_idx``, plus either the
        physical spatial columns (``z``, ``y``, ``x``) when using
        ``search_range_micrometers`` or the index-space columns
        (``z_idx``, ``y_idx``, ``x_idx``) when using
        ``search_range_pixels``.  Any additional columns are passed
        through.
    search_range_pixels : float or tuple of float, optional
        Per-axis maximum displacement in pixel (index) units between two
        consecutive frames.  A scalar is broadcast to every spatial
        axis; a 3-tuple is interpreted as ``(z, y, x)``.  Mutually
        exclusive with ``search_range_micrometers``.
    search_range_micrometers : float or tuple of float, optional
        Per-axis maximum displacement in micrometres between two
        consecutive frames.  A scalar is broadcast to every spatial
        axis; a 3-tuple is interpreted as ``(z, y, x)``.  Mutually
        exclusive with ``search_range_pixels``.
    memory : int
        Number of frames a trajectory is allowed to go unmatched
        before it is closed.  Default 0 (a trajectory must be matched
        in every frame to stay open).
    adaptive_step : float, optional
        If given, the search range for a track that has been unseen
        for ``k`` frames grows as ``search_range * sqrt(1 + k)``
        multiplied by ``adaptive_step`` (i.e. ``search_range *
        sqrt(1 + adaptive_step * k)``).  This is the Brownian-motion
        scaling: tracks that have been missing for a while are
        searched over a wider area.

    Returns
    -------
    polars.DataFrame
        A new DataFrame with the same columns and row order as ``locs``,
        plus a ``particle_id`` column (``Int32``).

    Notes
    -----
    The per-axis search ranges define an ellipsoidal gating region: a
    candidate link is accepted iff the normalized distance
    ``sqrt(sum((d_i / r_i)**2))`` is at most 1 (with
    ``adaptive_step`` scaling the threshold).  A scalar search range
    therefore reduces to the familiar Euclidean ball.
    """
    # Resolve the search-range mode (pixel or micrometre) and the
    # spatial columns it implies, then validate the remaining arguments.
    ranges, spatial_cols = _resolve_search_ranges(
        search_range_pixels, search_range_micrometers, locs
    )
    required = {"c", "t_idx"} | set(spatial_cols)
    missing = required - set(locs.columns)
    if missing:
        raise ValueError(f"locs is missing required columns: {sorted(missing)}")
    if memory < 0:
        raise ValueError(f"`memory` must be non-negative, got {memory}.")
    if adaptive_step is not None and adaptive_step < 0:
        raise ValueError(f"`adaptive_step` must be non-negative, got {adaptive_step}.")

    n_rows = locs.shape[0]
    if n_rows == 0:
        return locs.with_columns(polars.Series([], dtype=polars.Int32).alias("particle_id"))

    # Sort by (channel, frame) for sequential processing, but keep a
    # ``__row_index`` so we can return the result in the input order.
    sorted_locs = locs.with_row_index(name="__row_index").sort(["c", "t_idx"])

    t_idx_arr = sorted_locs["t_idx"].to_numpy().astype(np.int64, copy=False)
    c_arr = sorted_locs["c"].to_numpy()
    pos_cols = [sorted_locs[col].to_numpy().astype(np.float64, copy=False) for col in spatial_cols]
    idx_arr = sorted_locs["__row_index"].to_numpy()

    # Per-row particle id, in sorted order.
    particle_ids = np.empty(n_rows, dtype=np.int32)
    next_id = np.int32(0)

    # Split the sorted array by channel and process each in turn.  We
    # iterate over the sorted array once and slice out the chunks that
    # belong to each unique channel.
    unique_channels, channel_starts = np.unique(c_arr, return_index=True)
    channel_ends = np.append(channel_starts[1:], np.array(len(c_arr)).reshape(()))

    for chan_start, chan_end in zip(channel_starts, channel_ends):
        # Each track is a small dict: last position, last frame on which
        # it was matched, and the assigned particle id.
        active_tracks: list[dict] = []

        frame_start = chan_start
        while frame_start < chan_end:
            t = t_idx_arr[frame_start]
            frame_end = frame_start
            while frame_end < chan_end and t_idx_arr[frame_end] == t:
                frame_end += 1

            current_t = int(t)
            det_pos = np.stack(
                [col[frame_start:frame_end] for col in pos_cols],
                axis=1,
            )
            n_dets = frame_end - frame_start

            # Close every active track whose last match is too far in
            # the past.  ``memory`` counts the number of *empty* frames
            # a track is allowed to go through, so a track that was
            # last seen at ``last_seen_t`` is alive at ``current_t`` iff
            # ``current_t - last_seen_t - 1 <= memory``, i.e. iff the
            # number of intervening frames is at most ``memory``.
            if active_tracks:
                kept: list[dict] = []
                for tr in active_tracks:
                    if current_t - tr["last_seen_t"] <= memory + 1:
                        kept.append(tr)
                active_tracks = kept

            if not active_tracks:
                # No open tracks; every detection starts a new one.
                for j in range(n_dets):
                    pid = next_id
                    next_id += 1
                    particle_ids[frame_start + j] = pid
                    active_tracks.append(
                        {
                            "last_pos": det_pos[j],
                            "last_seen_t": current_t,
                            "particle_id": pid,
                        }
                    )
            elif n_dets > 0:
                # Build the cost matrix and solve the assignment.
                track_pos = np.stack([tr["last_pos"] for tr in active_tracks])
                diff = track_pos[:, None, :] - det_pos[None, :, :]

                # Normalized Euclidean (ellipsoidal) distance: each axis
                # is scaled by its per-axis search range, so a scalar
                # range reduces to the Euclidean ball and a per-axis
                # tuple gives an ellipsoid with semi-axes ``r_i``.
                norm_diff = diff / ranges[None, None, :]
                dist = np.sqrt(np.sum(norm_diff * norm_diff, axis=2))

                # Effective per-track threshold (Brownian-motion scaling:
                # sqrt growth in the number of unseen frames).
                if adaptive_step is not None:
                    frames_unseen = np.array(
                        [current_t - tr["last_seen_t"] for tr in active_tracks]
                    )
                    effective_threshold = np.sqrt(1.0 + adaptive_step * frames_unseen)
                else:
                    effective_threshold = np.ones(len(active_tracks), dtype=np.float64)

                cost = dist.copy()
                # Use a large finite value instead of ``np.inf`` for
                # out-of-range entries: ``scipy.optimize.linear_sum_assignment``
                # raises ``ValueError: cost matrix is infeasible`` when
                # the matrix contains ``inf`` (the algorithm treats ``inf``
                # entries as truly unreachable, which is not what we
                # want for a thresholded cost).
                big = 1.0e12
                cost[dist > effective_threshold[:, None]] = big

                if np.all(cost >= big):
                    row_ind = np.array([], dtype=np.int64)
                    col_ind = np.array([], dtype=np.int64)
                else:
                    row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost)

                matched_tracks: set[int] = set()
                matched_dets: set[int] = set()
                for i, j in zip(row_ind, col_ind):
                    # Filter out matches that fell back to the ``big``
                    # cost: those are out of range and should be treated
                    # as unmatched.
                    if cost[i, j] >= big:
                        continue
                    active_tracks[i]["last_pos"] = det_pos[j]
                    active_tracks[i]["last_seen_t"] = current_t
                    particle_ids[frame_start + j] = active_tracks[i]["particle_id"]
                    matched_tracks.add(int(i))
                    matched_dets.add(int(j))

                # Unmatched detections start new trajectories.
                for j in range(n_dets):
                    if j in matched_dets:
                        continue
                    pid = next_id
                    next_id += 1
                    particle_ids[frame_start + j] = pid
                    active_tracks.append(
                        {
                            "last_pos": det_pos[j],
                            "last_seen_t": current_t,
                            "particle_id": pid,
                        }
                    )

            frame_start = frame_end

    # Reorder particle_ids back to the input order.
    original_particle_ids = np.empty(n_rows, dtype=np.int32)
    original_particle_ids[idx_arr] = particle_ids

    return locs.with_columns(
        polars.Series(original_particle_ids, dtype=polars.Int32).alias("particle_id")
    )
