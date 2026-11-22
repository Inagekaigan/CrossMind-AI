"""Initiation labels from densely sampled crossing-state annotations, not intent scores."""
import numpy as np


def crossing_initiation_label(times, crossing, anchor, horizon=2.0, max_gap_seconds=0.1):
    """Return 0/1, or None for current crossing, censoring or missing annotations.

    anchor indexes the last observation. Require full coverage through the horizon
    for BOTH classes, avoiding unequal follow-up selection. Onset is the first
    annotated 0->1 transition; temporal precision equals annotation resolution.
    """
    times = np.asarray(times, float)
    crossing = np.asarray(crossing, float)
    if (times.ndim != 1 or not len(times) or crossing.shape != times.shape
            or not np.isfinite(times).all() or np.any(np.diff(times) <= 0)):
        raise ValueError("Aligned annotations and strictly increasing timestamps required")
    if not isinstance(anchor, (int, np.integer)) or not 0 <= anchor < len(times):
        raise ValueError("Invalid anchor")
    if not np.isfinite(horizon) or horizon <= 0 or not np.isfinite(max_gap_seconds) or max_gap_seconds <= 0:
        raise ValueError("Positive finite horizon and gap required")
    if np.any(~(np.isin(crossing, [0, 1]) | np.isnan(crossing))):
        raise ValueError("Crossing must be 0, 1, or NaN for unavailable")
    end = times[anchor] + horizon
    stop = int(np.searchsorted(times, end - 1e-9))
    if stop >= len(times) or crossing[anchor] != 0:
        return None
    segment = crossing[anchor:stop + 1]
    if not np.isfinite(segment).all() or np.any(np.diff(times[anchor:stop + 1]) > max_gap_seconds + 1e-9):
        return None
    future = crossing[anchor + 1:stop + 1][times[anchor + 1:stop + 1] <= end + 1e-9]
    return int(np.any(future == 1))
