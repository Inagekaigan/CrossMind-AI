"""Explicit per-window trajectory metrics; no fabricated distribution metrics."""
import numpy as np


def trajectory_metrics(prediction, target, miss_threshold):
    prediction, target = np.asarray(prediction, float), np.asarray(target, float)
    if (prediction.ndim != 3 or prediction.shape[0] < 1 or target.ndim != 2
            or target.shape[1] != 2 or len(target) == 0 or prediction.shape[1:] != target.shape):
        raise ValueError("Expected predictions [K,F,2], target [F,2], K,F > 0")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("Metrics require finite coordinates")
    if not np.isfinite(miss_threshold) or miss_threshold <= 0:
        raise ValueError("Miss threshold must be positive in the reported coordinate units")
    errors = np.linalg.norm(prediction - target[None], axis=-1)
    ade, fde = errors.mean(axis=1), errors[:, -1]
    return {"ADE_mode0": float(ade[0]), "FDE_mode0": float(fde[0]),
            "minADE": float(ade.min()), "minFDE": float(fde.min()),
            "miss": float(fde.min() > miss_threshold), "K": len(prediction)}
