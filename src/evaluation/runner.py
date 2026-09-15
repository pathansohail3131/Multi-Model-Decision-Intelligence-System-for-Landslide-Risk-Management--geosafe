"""Cross-validation execution: out-of-fold predictions and fold metrics.

Out-of-fold (OOF) predictions are the backbone of everything downstream. Each
sample's OOF probability comes from a model that never saw it in training, so
the same array can honestly serve three purposes: scoring a model, fitting a
combination rule over several models, and calibrating a model before its
disagreement with others is measured.

Any combination rule or calibrator fitted on in-sample predictions would be
fitted on values the base models had already memorised, which is how a stacked
ensemble ends up looking excellent in development and losing to its own best
member in evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.evaluation.metrics import METRIC_KEYS, compute_metrics
from src.evaluation.spatial_cv import SpatialBlockCV, random_cv_splits
from src.models.estimators import (
    apply_imbalance_strategy,
    make_estimator,
    prepare_features,
)
from src.utils.logging_setup import get_logger

__all__ = ["CVResult", "cross_validate_oof", "aggregate_fold_metrics"]

logger = get_logger(__name__)


@dataclass
class CVResult:
    """Outcome of one cross-validation run.

    Attributes
    ----------
    oof : numpy.ndarray
        Out-of-fold probability for every sample. NaN where a sample was never
        in a test fold (possible only if a fold came out empty).
    fold_metrics : list of dict
        Metrics per fold.
    summary : dict
        Mean and standard deviation of each metric across folds.
    scheme : str
        ``"spatial"`` or ``"random"``.
    label : str
        Human-readable description of the run.
    n_folds : int
        Number of folds actually executed.
    """

    oof: np.ndarray
    fold_metrics: list[dict] = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    scheme: str = "spatial"
    label: str = ""
    n_folds: int = 0

    def as_dict(self) -> dict:
        """Return a JSON-serialisable summary (excluding the OOF array)."""
        return {
            "label": self.label,
            "scheme": self.scheme,
            "n_folds": self.n_folds,
            "summary": self.summary,
            "fold_metrics": self.fold_metrics,
        }


def aggregate_fold_metrics(fold_metrics: list[dict]) -> dict:
    """Summarise per-fold metrics as mean and standard deviation.

    The standard deviation matters as much as the mean here. With 417 positives
    over 5 spatial folds, a fold holds roughly 80 positives, so fold-to-fold
    variation is substantial and a mean quoted alone overstates how precisely
    two models can be separated.

    Parameters
    ----------
    fold_metrics : list of dict
        Metrics from each fold.

    Returns
    -------
    dict
        ``<metric>_mean`` and ``<metric>_std`` for each reported metric.
    """
    out: dict[str, float] = {}
    for key in METRIC_KEYS:
        values = np.array(
            [m[key] for m in fold_metrics if key in m and np.isfinite(m[key])],
            dtype="float64",
        )
        if values.size:
            out[f"{key}_mean"] = float(values.mean())
            out[f"{key}_std"] = float(values.std(ddof=0))
    for key in ("tp", "fp", "fn", "tn"):
        out[f"{key}_total"] = int(sum(m.get(key, 0) for m in fold_metrics))
    return out


def cross_validate_oof(
    X: pd.DataFrame,
    y: np.ndarray,
    coords: np.ndarray,
    *,
    scheme: str = "spatial",
    n_splits: int = 5,
    block_size_m: float = 5000.0,
    kind: str = "catboost",
    imbalance: str = "none",
    categorical_features: list[str] | None = None,
    seed: int = 42,
    threshold: float = 0.5,
    label: str = "",
    estimator_overrides: dict | None = None,
) -> CVResult:
    """Run cross-validation and collect out-of-fold predictions.

    Parameters
    ----------
    X : pandas.DataFrame
        Encoded features.
    y : numpy.ndarray
        Binary target.
    coords : numpy.ndarray, shape (n, 2)
        Projected coordinates, required for spatial blocking.
    scheme : {"spatial", "random"}, default "spatial"
        Which splitter to use. ``"random"`` exists solely to quantify the
        optimism spatial structure injects and is never used for selection.
    n_splits : int, default 5
        Number of folds.
    block_size_m : float, default 5000.0
        Spatial block side length; ignored for the random scheme.
    kind : str, default "catboost"
        Estimator kind.
    imbalance : str, default "none"
        Class-imbalance strategy. Resampling is applied inside each fold.
    categorical_features : list of str, optional
        Categorical column names.
    seed : int, default 42
        Seed for splitting, resampling and the estimator.
    threshold : float, default 0.5
        Threshold for the threshold-dependent metrics. Stage 5 replaces this
        with a derived value; during Stage 3 and 4 the ranking metrics
        (AUC-PR, ROC-AUC) are what drive decisions, and those are
        threshold-free.
    label : str, optional
        Description used in logs and reports.
    estimator_overrides : dict, optional
        Hyperparameters overriding the estimator defaults.

    Returns
    -------
    CVResult
        OOF predictions, per-fold metrics and their summary.
    """
    y = np.asarray(y).astype(int)
    categorical_features = categorical_features or []
    overrides = estimator_overrides or {}

    if scheme == "spatial":
        splitter = SpatialBlockCV(
            n_splits=n_splits, block_size_m=block_size_m, random_state=seed
        )
        splits = list(splitter.split(X, y, coords))
    elif scheme == "random":
        splits = list(random_cv_splits(y, n_splits=n_splits, random_state=seed))
    else:
        raise ValueError(f"unknown CV scheme {scheme!r}; expected 'spatial' or 'random'")

    oof = np.full(len(y), np.nan, dtype="float64")
    fold_metrics: list[dict] = []

    for fold, (train_idx, test_idx) in enumerate(splits):
        y_train = y[train_idx]
        if len(np.unique(y_train)) < 2:
            logger.warning(
                "%s fold %d has a single class in training; skipping", label, fold
            )
            continue

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        # Resampling strictly inside the fold. Doing it before the split would
        # place synthetic copies of training points into the test set.
        X_train, y_train = apply_imbalance_strategy(
            X_train, y_train, imbalance, categorical_features, seed=seed
        )

        estimator = make_estimator(
            kind=kind,
            seed=seed,
            imbalance=imbalance,
            categorical_features=[c for c in categorical_features if c in X.columns],
            n_positive=int(y_train.sum()),
            n_negative=int(len(y_train) - y_train.sum()),
            **overrides,
        )

        prepared_train = prepare_features(X_train, kind, categorical_features)
        prepared_test = prepare_features(X_test, kind, categorical_features)
        # One-hot expansion (logreg) can yield different columns between train
        # and test when a level is absent from one side; align explicitly.
        prepared_test = prepared_test.reindex(
            columns=prepared_train.columns, fill_value=0
        )

        estimator.fit(prepared_train, y_train)
        oof[test_idx] = estimator.predict_proba(prepared_test)[:, 1]

        metrics = compute_metrics(y[test_idx], oof[test_idx], threshold=threshold)
        metrics["fold"] = fold
        fold_metrics.append(metrics)

    summary = aggregate_fold_metrics(fold_metrics)
    if summary:
        logger.info(
            "%-42s %-7s AUC-PR %.4f+-%.4f  ROC-AUC %.4f+-%.4f",
            label or kind,
            scheme,
            summary.get("average_precision_mean", float("nan")),
            summary.get("average_precision_std", float("nan")),
            summary.get("roc_auc_mean", float("nan")),
            summary.get("roc_auc_std", float("nan")),
        )

    return CVResult(
        oof=oof,
        fold_metrics=fold_metrics,
        summary=summary,
        scheme=scheme,
        label=label or f"{kind}/{imbalance}",
        n_folds=len(fold_metrics),
    )
