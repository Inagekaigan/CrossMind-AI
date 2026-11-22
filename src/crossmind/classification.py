"""Classification metrics with explicit undefined cases and calibration bins."""
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix


def classification_metrics(y, probability):
    y, p = np.asarray(y), np.asarray(probability)
    if y.shape != p.shape or y.ndim != 1 or not len(y) or not np.isin(y, [0, 1]).all():
        raise ValueError("Aligned nonempty binary targets required")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Probabilities outside [0,1]")
    hard = p >= .5
    both = len(np.unique(y)) == 2
    bins = np.minimum((p*10).astype(int), 9)
    ece = sum(np.mean(bins == b)*abs(p[bins == b].mean()-y[bins == b].mean())
              for b in range(10) if np.any(bins == b))
    return {"accuracy": float(accuracy_score(y, hard)),
            "precision": float(precision_score(y, hard, zero_division=0)),
            "recall": float(recall_score(y, hard, zero_division=0)),
            "f1": float(f1_score(y, hard, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, p)) if both else None,
            "average_precision": float(average_precision_score(y, p)) if both else None,
            "undefined_reason": None if both else "Only one target class present",
            "brier": float(np.mean((p-y)**2)), "ece_10_bins": float(ece),
            "confusion_matrix": confusion_matrix(y, hard, labels=[0, 1]).tolist(),
            "count": len(y), "positive": int(y.sum()), "threshold": .5,
            "zero_division_policy": "precision/recall/F1 return 0 when denominator is zero"}
