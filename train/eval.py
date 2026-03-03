"""Evaluation helpers for THORN with multi-class support."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


@dataclass
class ClassificationMetrics:
    pr_auc: float
    roc_auc: float
    precision_at_k: float
    accuracy: float


@dataclass
class DriftMetrics:
    routing_shift_l1: float
    routing_entropy_early: float
    routing_entropy_late: float
    collapse_score: float


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, frac_k: float) -> float:
    n = len(y_true)
    k = max(1, int(frac_k * n))
    top_idx = np.argsort(-y_score)[:k]
    return float(np.mean(y_true[top_idx]))


def classification_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    frac_k: float = 0.1,
    num_classes: int | None = None,
) -> ClassificationMetrics:
    """Compute classification metrics, handling both binary and multi-class cases.

    Args:
        y_true: [N] integer labels
        y_score: [N] scores for binary, or [N, C] probabilities for multi-class
        frac_k: fraction for precision@k
        num_classes: if provided, forces multi-class handling
    """
    n_unique = len(np.unique(y_true))
    is_multiclass = (num_classes is not None and num_classes > 2) or (n_unique > 2) or (y_score.ndim == 2 and y_score.shape[1] > 1)

    if is_multiclass and y_score.ndim == 2:
        # Multi-class: use macro-averaged metrics with OvR
        try:
            pr_auc = float(average_precision_score(y_true, y_score, average="macro"))
        except ValueError:
            pr_auc = 0.0

        try:
            roc_auc = float(roc_auc_score(y_true, y_score, multi_class="ovr", average="macro"))
        except ValueError:
            roc_auc = 0.0

        y_pred = y_score.argmax(axis=1)
        accuracy = float((y_true == y_pred).mean())

        # Precision@k: fraction of top-k by max predicted probability that are correct
        max_prob = y_score.max(axis=1)
        pak = precision_at_k(
            (y_true == y_pred).astype(np.int64),
            max_prob,
            frac_k,
        )

        return ClassificationMetrics(
            pr_auc=pr_auc,
            roc_auc=roc_auc,
            precision_at_k=pak,
            accuracy=accuracy,
        )
    else:
        # Binary case (original behavior)
        if y_score.ndim == 2:
            y_score = y_score[:, 1]
        y_pred = (y_score >= 0.5).astype(np.int64)
        return ClassificationMetrics(
            pr_auc=float(average_precision_score(y_true, y_score)),
            roc_auc=float(roc_auc_score(y_true, y_score)),
            precision_at_k=precision_at_k(y_true, y_score, frac_k=frac_k),
            accuracy=float((y_true == y_pred).mean()),
        )
