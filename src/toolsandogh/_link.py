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

from collections.abc import Callable, Sequence

import numpy as np
import polars
import scipy.optimize


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


def link(
    locs: polars.DataFrame,
    *,
    search_range_pixels: float | int | Sequence[float | int] | None = None,
    search_range_micrometers: float | int | Sequence[float | int] | None = None,
    memory: int = 0,
    adaptive_step: float | None = None,
    on_progress: Callable[[int], None] | None = None,
) -> polars.DataFrame:
    """
    Link per-frame emitter localizations into trajectories.

    For every detected frame, the still-open trajectories are matched
    to the new detections with the Hungarian algorithm.  Each detection
    is either appended to an existing trajectory (if it is within the
    search range of the trajectory's last known position) or starts a
    new one.  A trajectory that goes unmatched for more than
    ``memory`` frames is closed.

    Each channel (``c`` column) is linked independently.  When ``c`` is
    absent all rows are linked together.  Frames are counted by the
    integer ``t_idx`` column, so ``memory`` and any adaptive scaling are
    independent of the frame rate.

    Parameters
    ----------
    locs : polars.DataFrame
        Per-frame emitter table, typically from
        :func:`toolsandogh.locate`.  Must contain ``t_idx`` and either
        the physical spatial columns (``z``, ``y``, ``x``) when using
        ``search_range_micrometers`` or the index-space columns
        (``z_idx``, ``y_idx``, ``x_idx``) when using
        ``search_range_pixels``.  The channel column ``c`` is optional;
        when present, each channel is linked independently.  Any
        additional columns are passed through.
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
    on_progress : callable, optional
        A callback invoked with the number of rows (detections)
        processed so far.  It is called once with ``0`` before any row
        is processed, and again after one or more rows have been
        processed.  The final value equals ``locs.height``.  ``None``
        (the default) disables progress reporting.

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
    # Validate the search-range arguments.  Exactly one of the two
    # mutually-exclusive modes must be supplied; it determines which
    # spatial columns are read and required.
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
        columns = ("z", "y", "x")
        search_ranges = _parse_search_range(
            search_range_micrometers, param_name="search_range_micrometers"
        )
    else:
        assert search_range_pixels is not None
        columns = ("z_idx", "y_idx", "x_idx")
        search_ranges = _parse_search_range(search_range_pixels, param_name="search_range_pixels")

    # Validate the remaining arguments and the input schema.  ``c`` is
    # optional; when absent all rows are linked together.
    required = {"t_idx"} | set(columns)
    missing = required - set(locs.columns)
    if missing:
        raise ValueError(f"locs is missing required columns: {sorted(missing)}")
    if memory < 0:
        raise ValueError(f"`memory` must be non-negative, got {memory}.")
    if adaptive_step is not None and adaptive_step < 0:
        raise ValueError(f"`adaptive_step` must be non-negative, got {adaptive_step}.")
    if on_progress is not None and not callable(on_progress):
        raise TypeError(f"`on_progress` must be callable, got {type(on_progress).__name__}.")

    n_rows = locs.height
    if n_rows == 0:
        if on_progress is not None:
            on_progress(0)
        return locs.with_columns(polars.Series([], dtype=polars.Int32).alias("particle_id"))

    if on_progress is not None:
        on_progress(0)

    # Attach a row index so we can write particle ids back into the
    # original row order after sorting and grouping.  Sort once by the
    # full grouping key so the ``partition_by`` calls below do not need
    # to re-sort: rows within each group are already in frame order.
    has_channel = "c" in locs.columns
    sort_keys = ["c", "t_idx"] if has_channel else ["t_idx"]
    indexed = locs.with_row_index(name="__row_index").sort(sort_keys)

    # Per-row particle id, indexed by the original row order.
    particle_ids = np.full(n_rows, -1, dtype=np.int32)
    next_id = np.int32(0)
    rows_done = 0

    # Outer loop: one iteration per channel (or a single pass over the
    # whole table when ``c`` is absent).  Each channel keeps its own set
    # of active tracks, which is reset at the top of the loop.
    channel_groups = indexed.partition_by("c", maintain_order=True) if has_channel else [indexed]
    for channel_df in channel_groups:
        active_tracks: list[dict] = []

        # Inner loop: one iteration per detected frame, in ascending
        # ``t_idx`` order.  ``partition_by`` returns one DataFrame per
        # frame, carrying the original row indices.
        for frame_df in channel_df.partition_by("t_idx", maintain_order=True):
            frame_t = int(frame_df["t_idx"][0])
            row_indices = frame_df["__row_index"].to_numpy()
            det_pos = np.stack(
                [frame_df[col].to_numpy().astype(np.float64, copy=False) for col in columns],
                axis=1,
            )
            n_dets = frame_df.height

            # Close every active track whose last match is too far in
            # the past.  ``memory`` counts the number of *empty* frames
            # a track is allowed to go through, so a track that was
            # last seen at ``last_seen_t`` is alive at ``frame_t`` iff
            # ``frame_t - last_seen_t - 1 <= memory``.
            if active_tracks:
                kept: list[dict] = []
                for tr in active_tracks:
                    if frame_t - tr["last_seen_t"] <= memory + 1:
                        kept.append(tr)
                active_tracks = kept

            if not active_tracks:
                # No open tracks; every detection starts a new one.
                for j in range(n_dets):
                    pid = next_id
                    next_id += 1
                    particle_ids[row_indices[j]] = pid
                    active_tracks.append(
                        {
                            "last_pos": det_pos[j],
                            "last_seen_t": frame_t,
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
                norm_diff = diff / search_ranges[None, None, :]
                dist = np.sqrt(np.sum(norm_diff * norm_diff, axis=2))

                # Effective per-track threshold (Brownian-motion
                # scaling: sqrt growth in the number of unseen frames).
                if adaptive_step is not None:
                    frames_unseen = np.array([frame_t - tr["last_seen_t"] for tr in active_tracks])
                    effective_threshold = np.sqrt(1.0 + adaptive_step * frames_unseen)
                else:
                    effective_threshold = np.ones(len(active_tracks), dtype=np.float64)

                cost = dist.copy()
                # Use a large finite value instead of ``np.inf`` for
                # out-of-range entries:
                # ``scipy.optimize.linear_sum_assignment`` raises
                # ``ValueError: cost matrix is infeasible`` when the
                # matrix contains ``inf`` (the algorithm treats ``inf``
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
                    active_tracks[i]["last_seen_t"] = frame_t
                    particle_ids[row_indices[j]] = active_tracks[i]["particle_id"]
                    matched_tracks.add(int(i))
                    matched_dets.add(int(j))

                # Unmatched detections start new trajectories.
                for j in range(n_dets):
                    if j in matched_dets:
                        continue
                    pid = next_id
                    next_id += 1
                    particle_ids[row_indices[j]] = pid
                    active_tracks.append(
                        {
                            "last_pos": det_pos[j],
                            "last_seen_t": frame_t,
                            "particle_id": pid,
                        }
                    )

            # Report progress.
            rows_done += n_dets
            if on_progress is not None:
                on_progress(rows_done)

    return locs.with_columns(polars.Series(particle_ids, dtype=polars.Int32).alias("particle_id"))
