"""Timestamp-aware deterministic baselines. Coordinates retain their input units."""
import numpy as np


def validate_history(positions, times):
    positions, times = np.asarray(positions, dtype=float), np.asarray(times, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 2 or times.shape != (len(positions),):
        raise ValueError("Expected positions [T,2] and timestamps [T]")
    if len(times) < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("At least two finite, strictly increasing timestamps required")
    if not np.isfinite(positions).all():
        raise ValueError("Missing positions must be filtered explicitly before prediction")
    return positions, times


def predict(positions, times, future_times, method="constant_velocity", max_gap_seconds=0.5):
    """Fit least-squares velocity; anchor at last observation. Never use future positions.

    Reject interruptions longer than max_gap_seconds; callers must start a new history.
    Returns [1,F,2]: one deterministic mode, with no implied mode probability.
    """
    positions, times = validate_history(positions, times)
    future = np.asarray(future_times, dtype=float)
    if (future.ndim != 1 or not len(future) or not np.isfinite(future).all()
            or np.any(np.diff(future) <= 0) or future[0] <= times[-1]):
        raise ValueError("Forecast timestamps must increase strictly after observations")
    if not np.isfinite(max_gap_seconds) or max_gap_seconds <= 0:
        raise ValueError("max_gap_seconds must be positive and finite")
    if np.any(np.diff(times) > max_gap_seconds + 1e-9):
        raise ValueError("Track interruption exceeds configured gap limit")
    if method == "constant_velocity":
        centered = times - times.mean()
        velocity = (centered[:, None] * (positions - positions.mean(axis=0))).sum(axis=0)
        velocity /= np.sum(centered ** 2)
    elif method == "stationary":
        velocity = np.zeros(2)
    else:
        raise ValueError(f"Unknown baseline: {method}")
    return (positions[-1] + (future - times[-1])[:, None] * velocity)[None]
