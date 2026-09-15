"""Estimator construction and class-imbalance handling.

One algorithm is used across all five models — the four thematic models and the
full-feature benchmark — and that uniformity is deliberate. The claim this
architecture rests on is that *diversity from data views* succeeds where
*diversity from algorithms* failed. Varying the feature subset and the learner
at once would confound the two and make the claim untestable: if the ensemble
won, there would be no way to say which change caused it.

A per-theme learner comparison is still run in Stage 4, as an explicitly
reported deviation rather than a default, so the question is answered with
evidence instead of assertion.

CatBoost is the default learner for three reasons:

1. It led the previous phase's comparison (0.9329 ROC-AUC).
2. It consumes ``lithology`` and ``LULC`` as genuine categoricals, so no
   integer code is ever read as an ordering.
3. Its ordered boosting is the right defence for very sparse categorical
   levels — LULC built-up carries 21 positives in 45 samples, which a
   naive target encoder would memorise outright.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.utils.logging_setup import get_logger

__all__ = [
    "make_estimator",
    "prepare_features",
    "apply_imbalance_strategy",
    "IMBALANCE_STRATEGIES",
    "ESTIMATOR_KINDS",
]

logger = get_logger(__name__)

#: Strategies compared empirically in Stage 3.
IMBALANCE_STRATEGIES = ("none", "class_weight", "scale_pos_weight", "smote_in_fold")

#: Learners available to the pipeline.
ESTIMATOR_KINDS = ("catboost", "xgboost", "logreg")

#: Conservative defaults. Depth is kept modest because the positive class has
#: only 417 members: a deep tree on 417 positives memorises rather than
#: generalises, and spatial CV would expose that as a large train/test gap.
_CATBOOST_DEFAULTS: dict[str, Any] = {
    "iterations": 600,
    "depth": 6,
    "learning_rate": 0.05,
    "l2_leaf_reg": 3.0,
    "bootstrap_type": "Bernoulli",
    "subsample": 0.8,
    "verbose": 0,
    "allow_writing_files": False,
}

_XGBOOST_DEFAULTS: dict[str, Any] = {
    "n_estimators": 600,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "enable_categorical": True,
    "eval_metric": "aucpr",
    "verbosity": 0,
}


def prepare_features(
    X: pd.DataFrame, kind: str, categorical_features: list[str]
) -> pd.DataFrame:
    """Coerce encoded features into the dtypes a given learner expects.

    Parameters
    ----------
    X : pandas.DataFrame
        Encoded features from :class:`~src.features.encoders.FeatureEncoder`.
    kind : str
        Estimator kind: ``"catboost"``, ``"xgboost"`` or ``"logreg"``.
    categorical_features : list of str
        Names of the categorical columns present in ``X``.

    Returns
    -------
    pandas.DataFrame
        A copy with dtypes appropriate to the learner.

    Notes
    -----
    Each learner wants a different representation of the same nominal column —
    CatBoost takes strings, XGBoost takes a pandas ``category``, and a linear
    model needs one-hot — so this conversion lives in one place rather than
    being repeated at every call site where it could drift.
    """
    present = [c for c in categorical_features if c in X.columns]
    out = X.copy()

    if kind == "catboost":
        # CatBoost requires str or int for categorical columns and rejects the
        # pandas nullable StringDtype, so materialise plain python strings.
        for column in present:
            out[column] = out[column].astype("object").fillna("missing").astype(str)
    elif kind == "xgboost":
        for column in present:
            out[column] = out[column].astype("object").fillna("missing").astype(
                "category"
            )
    elif kind == "logreg":
        if present:
            out = pd.get_dummies(out, columns=present, dummy_na=False, dtype="float64")
    else:
        raise ValueError(f"unknown estimator kind: {kind!r}")

    return out


def make_estimator(
    kind: str = "catboost",
    seed: int = 42,
    imbalance: str = "none",
    categorical_features: list[str] | None = None,
    n_positive: int | None = None,
    n_negative: int | None = None,
    **overrides: Any,
):
    """Construct an estimator wired for the chosen imbalance strategy.

    Parameters
    ----------
    kind : str, default "catboost"
        One of :data:`ESTIMATOR_KINDS`.
    seed : int, default 42
        Random seed, passed explicitly rather than inherited from a global RNG.
    imbalance : str, default "none"
        One of :data:`IMBALANCE_STRATEGIES`. ``"smote_in_fold"`` is handled by
        :func:`apply_imbalance_strategy` at fit time, not here, so the
        estimator itself is built unweighted for that arm.
    categorical_features : list of str, optional
        Categorical column names, needed by CatBoost.
    n_positive, n_negative : int, optional
        Class counts, needed to derive ``scale_pos_weight``.
    **overrides
        Hyperparameters overriding the defaults.

    Returns
    -------
    estimator
        An unfitted scikit-learn-compatible classifier.

    Raises
    ------
    ValueError
        On an unknown kind or imbalance strategy.
    """
    if kind not in ESTIMATOR_KINDS:
        raise ValueError(f"unknown estimator kind {kind!r}; expected {ESTIMATOR_KINDS}")
    if imbalance not in IMBALANCE_STRATEGIES:
        raise ValueError(
            f"unknown imbalance strategy {imbalance!r}; expected {IMBALANCE_STRATEGIES}"
        )

    categorical_features = categorical_features or []
    ratio = (
        float(n_negative) / float(n_positive)
        if (n_positive and n_negative and n_positive > 0)
        else 1.0
    )

    if kind == "catboost":
        from catboost import CatBoostClassifier

        params = dict(_CATBOOST_DEFAULTS)
        params["random_seed"] = seed
        if categorical_features:
            params["cat_features"] = list(categorical_features)
        if imbalance == "scale_pos_weight":
            params["scale_pos_weight"] = ratio
        elif imbalance == "class_weight":
            params["auto_class_weights"] = "Balanced"
        params.update(overrides)
        return CatBoostClassifier(**params)

    if kind == "xgboost":
        from xgboost import XGBClassifier

        params = dict(_XGBOOST_DEFAULTS)
        params["random_state"] = seed
        if imbalance == "scale_pos_weight":
            params["scale_pos_weight"] = ratio
        elif imbalance == "class_weight":
            # XGBoost has no class_weight; scale_pos_weight is the equivalent
            # lever, so the two arms coincide for this learner. Recorded here
            # so the ablation table is not read as two independent results.
            params["scale_pos_weight"] = ratio
        params.update(overrides)
        return XGBClassifier(**params)

    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    params: dict[str, Any] = {
        "max_iter": 2000,
        "random_state": seed,
        "class_weight": "balanced" if imbalance in ("class_weight", "scale_pos_weight") else None,
    }
    params.update(overrides)
    # The scaler is inside the pipeline, never beside it, so the model cannot
    # be applied without it.
    return Pipeline(
        [("scale", StandardScaler()), ("clf", LogisticRegression(**params))]
    )


def apply_imbalance_strategy(
    X: pd.DataFrame,
    y: np.ndarray,
    strategy: str,
    categorical_features: list[str],
    seed: int = 42,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Resample the training data if the strategy calls for it.

    Parameters
    ----------
    X : pandas.DataFrame
        Training features **for this fold only**.
    y : numpy.ndarray
        Training target for this fold.
    strategy : str
        One of :data:`IMBALANCE_STRATEGIES`.
    categorical_features : list of str
        Categorical column names, required by SMOTE-NC.
    seed : int, default 42
        Seed for the synthetic sampler.

    Returns
    -------
    X_out, y_out
        Possibly resampled training data.

    Notes
    -----
    Resampling happens **inside the fold**, never before splitting. Oversampling
    before the split places synthetic copies of training points into the test
    set, which inflates every metric and is one of the most common silent
    errors in imbalanced-classification write-ups.

    SMOTE-NC is used rather than plain SMOTE because two features are nominal;
    interpolating a "rock type" halfway between metamorphic and basalt would be
    meaningless. Even so, synthetic minority points interpolate in *feature*
    space, and a synthetic sample between two real landslides on different
    slopes need not correspond to any real terrain — which is why this arm is
    compared empirically rather than assumed helpful.
    """
    if strategy != "smote_in_fold":
        return X, y

    from imblearn.over_sampling import SMOTENC, SMOTE

    present = [c for c in categorical_features if c in X.columns]
    X_work = X.copy()
    for column in present:
        X_work[column] = X_work[column].astype("object").fillna("missing").astype(str)

    n_positive = int(np.asarray(y).sum())
    # SMOTE needs at least k_neighbors+1 minority samples in the fold.
    k = min(5, max(1, n_positive - 1))

    try:
        if present:
            indices = [X_work.columns.get_loc(c) for c in present]
            sampler = SMOTENC(
                categorical_features=indices, random_state=seed, k_neighbors=k
            )
        else:
            sampler = SMOTE(random_state=seed, k_neighbors=k)
        X_res, y_res = sampler.fit_resample(X_work, y)
    except ValueError as exc:
        logger.warning(
            "SMOTE failed on this fold (%s); falling back to unresampled data", exc
        )
        return X, y

    X_res = pd.DataFrame(X_res, columns=X_work.columns)
    # fit_resample returns object dtype for everything; restore numerics.
    for column in X_res.columns:
        if column not in present:
            X_res[column] = pd.to_numeric(X_res[column], errors="coerce")
    logger.debug(
        "SMOTE: %d -> %d rows (%d -> %d positive)",
        len(X),
        len(X_res),
        n_positive,
        int(np.asarray(y_res).sum()),
    )
    return X_res, np.asarray(y_res)
