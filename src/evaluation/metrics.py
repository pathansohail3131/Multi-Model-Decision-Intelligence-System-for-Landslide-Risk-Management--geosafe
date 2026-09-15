"""The metric suite.

AUC-PR (average precision) leads, not ROC-AUC. At 1:10.8 imbalance the ROC
curve is dominated by the majority class: a model can post a strong ROC-AUC
while having no operating point with usable precision, because the false
positive rate is diluted by 4,519 negatives. The precision-recall curve is
sensitive to exactly the regime a planner cares about.

ROC-AUC is still reported, for two reasons: it is what the previous phase
reported (0.9329 for CatBoost), so comparability matters, and it is what the
landslide-susceptibility literature quotes almost universally.

Threshold-dependent metrics (F1, recall, precision, kappa) are reported at an
explicit threshold that is never silently 0.5 — see ``docs/methodology.md``
§4.4.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

__all__ = [
    "compute_metrics",
    "threshold_candidates",
    "cost_minimising_threshold",
    "METRIC_KEYS",
]

#: Metrics reported everywhere, in a fixed order so tables stay comparable.
METRIC_KEYS = (
    "roc_auc",
    "average_precision",
    "f1",
    "f2",
    "recall",
    "precision",
    "cohen_kappa",
)


def compute_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """Compute the full metric suite for one set of predictions.

    Parameters
    ----------
    y_true : numpy.ndarray
        Binary ground truth.
    y_prob : numpy.ndarray
        Predicted probability of the positive class.
    threshold : float, default 0.5
        Operating threshold for the threshold-dependent metrics. The default is
        present only so the function is callable during development; the
        pipeline always passes a derived threshold.

    Returns
    -------
    dict
        Metric name mapped to value, plus the confusion-matrix counts and the
        share of samples predicted positive.

    Notes
    -----
    ``flagged_rate`` is included because a threshold that flags most of the
    district is useless to a planner whatever its recall, and that failure is
    invisible in recall, F1 or kappa alone.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype="float64")
    y_pred = (y_prob >= threshold).astype(int)

    # A degenerate fold with one class present cannot support ranking metrics.
    both_classes = len(np.unique(y_true)) > 1

    tn, fp, fn, tp = (
        confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        if both_classes or len(y_true)
        else (0, 0, 0, 0)
    )

    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if both_classes else float("nan"),
        "average_precision": float(average_precision_score(y_true, y_prob))
        if both_classes
        else float("nan"),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "f2": float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "threshold": float(threshold),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "flagged_rate": float(y_pred.mean()),
        "n": int(len(y_true)),
        "n_positive": int(y_true.sum()),
    }


def cost_minimising_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cost_fn: float = 10.0,
    cost_fp: float = 1.0,
) -> tuple[float, float]:
    """Find the threshold minimising expected misclassification cost.

    Parameters
    ----------
    y_true : numpy.ndarray
        Binary ground truth.
    y_prob : numpy.ndarray
        Predicted probability of the positive class.
    cost_fn : float, default 10.0
        Cost of a false negative — a missed landslide zone.
    cost_fp : float, default 1.0
        Cost of a false positive — an unnecessary survey visit.

    Returns
    -------
    threshold : float
        The cost-minimising threshold.
    cost : float
        Total cost at that threshold.

    Notes
    -----
    This is the honest way to set an operating point for an asymmetric
    problem. 0.5 is the decision boundary of a *balanced* problem with
    *symmetric* costs, and this problem is neither. The cost ratio is a stated
    assumption rather than a measured quantity, so the pipeline also reports
    how sensitive the chosen threshold is to that assumption.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype="float64")

    # Evaluate at every distinct predicted probability: the optimum always
    # sits at one of them, so a grid search would only add error.
    candidates = np.unique(y_prob)
    if candidates.size > 2000:
        candidates = np.quantile(candidates, np.linspace(0, 1, 2000))

    best_threshold, best_cost = 0.5, np.inf
    for threshold in candidates:
        y_pred = y_prob >= threshold
        fn = int(((y_true == 1) & ~y_pred).sum())
        fp = int(((y_true == 0) & y_pred).sum())
        cost = cost_fn * fn + cost_fp * fp
        if cost < best_cost:
            best_threshold, best_cost = float(threshold), float(cost)
    return best_threshold, best_cost


def threshold_candidates(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cost_fn: float = 10.0,
    cost_fp: float = 1.0,
    precision_floor: float = 0.25,
) -> dict[str, float]:
    """Derive several candidate operating thresholds for comparison.

    Reporting alternatives alongside the selected threshold is what turns a
    choice into a justified choice: the reader can see what a different
    objective would have picked and how far apart they are.

    Parameters
    ----------
    y_true : numpy.ndarray
        Binary ground truth.
    y_prob : numpy.ndarray
        Predicted probability of the positive class.
    cost_fn, cost_fp : float
        Misclassification costs for the cost-minimising threshold.
    precision_floor : float, default 0.25
        Minimum acceptable precision for the recall-maximising-subject-to-
        precision candidate.

    Returns
    -------
    dict
        Candidate name mapped to threshold. Always contains
        ``cost_minimising``, ``f2_optimal``, ``youden``, ``precision_floor``
        and ``default_half``.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype="float64")

    out: dict[str, float] = {}

    threshold, _ = cost_minimising_threshold(y_true, y_prob, cost_fn, cost_fp)
    out["cost_minimising"] = threshold

    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_prob)
    # precision_recall_curve returns one fewer threshold than points.
    precision, recall = precision[:-1], recall[:-1]

    with np.errstate(invalid="ignore", divide="ignore"):
        f2 = 5 * precision * recall / (4 * precision + recall)
    f2 = np.nan_to_num(f2, nan=0.0)
    out["f2_optimal"] = float(pr_thresholds[int(np.argmax(f2))])

    fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)
    out["youden"] = float(roc_thresholds[int(np.argmax(tpr - fpr))])

    ok = precision >= precision_floor
    # Among thresholds meeting the precision floor, take the one with the
    # highest recall. If none does, fall back to the best-precision point and
    # let the reported precision expose that the floor was unreachable.
    if ok.any():
        idx = int(np.argmax(np.where(ok, recall, -np.inf)))
    else:
        idx = int(np.argmax(precision))
    out["precision_floor"] = float(pr_thresholds[idx])

    out["default_half"] = 0.5
    return out
