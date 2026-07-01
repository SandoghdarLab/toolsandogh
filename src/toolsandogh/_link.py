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
"""

import numpy as np
import polars
import scipy.optimize


def link(
    locs: polars.DataFrame,
    *,
    search_range: float,
    memory: int = 0,
    adaptive_step: float | None = None,
) -> polars.DataFrame:
    """
    Link per-frame emitter localizations into trajectories.

    For every detected frame, the still-open trajectories are matched
    to the new detections with the Hungarian algorithm.  Each detection
    is either appended to an existing trajectory (if it is within
    ``search_range`` of the trajectory's last known position) or starts
    a new one.  A trajectory that goes unmatched for more than
    ``memory`` frames is closed.

    Each channel (``c`` column) is linked independently.

    Parameters
    ----------
    locs : polars.DataFrame
        Per-frame emitter table from :func:`toolsandogh.locate`.  Must
        contain the columns ``t``, ``c``, ``z``, ``y``, ``x``,
        ``contrast``.  Any additional columns are passed through.
    search_range : float
        Maximum Euclidean distance (in pixel units) an emitter may
        travel between two consecutive frames and still be linked.
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
    """
    # Validate the input schema.
    required = {"t", "c", "z", "y", "x", "contrast"}
    missing = required - set(locs.columns)
    if missing:
        raise ValueError(f"locs is missing required columns: {sorted(missing)}")
    if memory < 0:
        raise ValueError(f"`memory` must be non-negative, got {memory}.")
    if search_range <= 0:
        raise ValueError(f"`search_range` must be positive, got {search_range}.")
    if adaptive_step is not None and adaptive_step < 0:
        raise ValueError(f"`adaptive_step` must be non-negative, got {adaptive_step}.")

    n_rows = locs.shape[0]
    if n_rows == 0:
        return locs.with_columns(polars.Series([], dtype=polars.Int32).alias("particle_id"))

    # Sort by (channel, frame) for sequential processing, but keep a
    # ``__row_index`` so we can return the result in the input order.
    sorted_locs = locs.with_row_index(name="__row_index").sort(["c", "t"])

    t_arr = sorted_locs["t"].to_numpy()
    c_arr = sorted_locs["c"].to_numpy()
    z_arr = sorted_locs["z"].to_numpy()
    y_arr = sorted_locs["y"].to_numpy()
    x_arr = sorted_locs["x"].to_numpy()
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
        # Each track is a small dict: last (z, y, x) position, last
        # frame on which it was matched, and the assigned particle id.
        active_tracks: list[dict] = []

        frame_start = chan_start
        while frame_start < chan_end:
            t = t_arr[frame_start]
            frame_end = frame_start
            while frame_end < chan_end and t_arr[frame_end] == t:
                frame_end += 1

            current_t = int(t)
            det_pos = np.stack(
                [
                    z_arr[frame_start:frame_end],
                    y_arr[frame_start:frame_end],
                    x_arr[frame_start:frame_end],
                ],
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
                dist = np.sqrt(np.sum(diff * diff, axis=2))

                # Effective per-track search range (Brownian-motion
                # scaling: sqrt growth in the number of unseen frames).
                if adaptive_step is not None:
                    frames_unseen = np.array(
                        [current_t - tr["last_seen_t"] for tr in active_tracks]
                    )
                    effective_range = search_range * np.sqrt(1.0 + adaptive_step * frames_unseen)
                else:
                    effective_range = np.full(len(active_tracks), search_range, dtype=np.float64)

                cost = dist.copy()
                # Use a large finite value instead of ``np.inf`` for
                # out-of-range entries: ``scipy.optimize.linear_sum_assignment``
                # raises ``ValueError: cost matrix is infeasible`` when
                # the matrix contains ``inf`` (the algorithm treats ``inf``
                # entries as truly unreachable, which is not what we
                # want for a thresholded cost).
                big = 1.0e12
                cost[dist > effective_range[:, None]] = big

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
