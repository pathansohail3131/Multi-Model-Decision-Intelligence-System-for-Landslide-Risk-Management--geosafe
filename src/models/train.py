"""Stage 4: thematic models, error decorrelation, and the combination ablation.

Three questions are answered here, all from measured numbers:

1. **How strong is each thematic model on its own?** Including the geological
   model, which is expected to be weakest — it sees two factors, one of which
   carries a coverage gap.

2. **Do the thematic models actually make decorrelated errors?** This is the
   claim the whole architecture rests on. If their out-of-fold predictions
   correlate as tightly as five same-feature models would, the thematic split
   has not delivered what it was built for and the confidence raster is not
   meaningful. The correlation matrix is reported either way.

3. **Which combination rule should be used — if any?** The full-feature model
   alone is one of the candidates, so "no ensemble" can win. Stage 4 also runs
   a per-theme learner comparison, which tests whether letting each theme
   choose its own algorithm beats holding the algorithm fixed.

All out-of-fold predictions come from one identical spatial fold structure, so
the rules are compared on exactly the same data. Any combination rule with
parameters is fitted inside the folds, never on the out-of-fold matrix it is
then scored against.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.evaluation.metrics import compute_metrics
from src.evaluation.runner import aggregate_fold_metrics, cross_validate_oof
from src.evaluation.spatial_cv import SpatialBlockCV
from src.features.build import prepare_dataset
from src.features.themes import BENCHMARK_NAME
from src.models.bundle import ModelBundle, save_bundle
from src.models.combiners import COMBINERS, combine
from src.models.estimators import make_estimator, prepare_features
from src.utils.logging_setup import get_logger

__all__ = ["run_training_stage", "OOF_FILENAME"]

logger = get_logger(__name__)

#: Out-of-fold prediction matrix, consumed by stages 5 and 6.
OOF_FILENAME = "oof_predictions.csv"


def _load_stage3_choices(cfg) -> tuple[float, str]:
    """Read the block size and imbalance strategy Stage 3 selected.

    Falls back to configured defaults with a warning if Stage 3 has not run,
    so that a single stage can still be executed in isolation during
    development.
    """
    path = cfg.reports_dir / "spatial_cv.json"
    if not path.exists():
        logger.warning(
            "Stage 3 report not found at %s; falling back to configured defaults. "
            "Run stage 3 first for empirically selected values.",
            path,
        )
        return 5000.0, "none"
    with path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    block = float(report.get("selected_block_size_m", 5000.0))
    strategy = str(report.get("selected_imbalance_strategy", "none"))
    logger.info(
        "Using Stage 3 selections: block_size=%.0f m, imbalance=%r", block, strategy
    )
    return block, strategy


def _meta_learner_cv(
    P: np.ndarray,
    y: np.ndarray,
    coords: np.ndarray,
    block_size_m: float,
    n_splits: int,
    seed: int,
) -> np.ndarray:
    """Out-of-fold predictions for a logistic meta-learner over thematic OOF.

    The meta-learner is refitted inside each spatial fold. Fitting it once on
    the whole out-of-fold matrix and scoring it on that same matrix would be
    optimistic — the meta-learner would be tuned on the very predictions it is
    then judged against, which is a milder version of the error that made the
    previous stacked ensemble look better in development than it was.

    Parameters
    ----------
    P : numpy.ndarray, shape (n_samples, n_themes)
        Thematic out-of-fold probabilities.
    y : numpy.ndarray
        Binary target.
    coords : numpy.ndarray
        Projected coordinates.
    block_size_m : float
        Spatial block side length.
    n_splits : int
        Number of folds.
    seed : int
        Seed.

    Returns
    -------
    numpy.ndarray
        Meta-learner out-of-fold probabilities.
    """
    from sklearn.linear_model import LogisticRegression

    out = np.full(len(y), np.nan, dtype="float64")
    cv = SpatialBlockCV(n_splits=n_splits, block_size_m=block_size_m, random_state=seed)
    for train_idx, test_idx in cv.split(None, y, coords):
        if len(np.unique(y[train_idx])) < 2:
            continue
        model = LogisticRegression(max_iter=1000, random_state=seed)
        model.fit(P[train_idx], y[train_idx])
        out[test_idx] = model.predict_proba(P[test_idx])[:, 1]
    return out


def run_training_stage(cfg) -> dict:
    """Stage 4 entry point.

    Parameters
    ----------
    cfg : Config
        Resolved project configuration.

    Returns
    -------
    dict
        Report written to ``outputs/reports/models_ablation.json``.
    """
    dataset = prepare_dataset(cfg)
    seed = cfg.seed
    n_splits = int(cfg.get("cross_validation", "n_splits", default=5))
    kind = str(cfg.get("models", "kind", default="catboost"))
    block_size_m, imbalance = _load_stage3_choices(cfg)

    themes = list(cfg.theme_names)
    all_models = themes + [BENCHMARK_NAME]

    report: dict = {
        "block_size_m": block_size_m,
        "imbalance_strategy": imbalance,
        "estimator_kind": kind,
        "n_splits": n_splits,
        "seed": seed,
        "themes": themes,
    }

    # ------------------------------------------------------------------
    # 1. per-model spatial CV, sharing one fold structure
    # ------------------------------------------------------------------
    logger.info("--- individual model performance (spatial block CV) ---")
    oof: dict[str, np.ndarray] = {}
    individual: dict[str, dict] = {}

    for name in all_models:
        result = cross_validate_oof(
            dataset.theme_X(name),
            dataset.y,
            dataset.coords,
            scheme="spatial",
            n_splits=n_splits,
            block_size_m=block_size_m,
            kind=kind,
            imbalance=imbalance,
            categorical_features=dataset.theme_categoricals(name),
            seed=seed,
            label=f"{name} ({len(dataset.theme_features[name])} features)",
        )
        oof[name] = result.oof
        individual[name] = {
            "n_features": len(dataset.theme_features[name]),
            "features": dataset.theme_features[name],
            **result.as_dict(),
        }
    report["individual_models"] = individual

    # ------------------------------------------------------------------
    # 2. error decorrelation — the architecture's central claim
    # ------------------------------------------------------------------
    logger.info("--- cross-theme error decorrelation ---")
    P = np.column_stack([oof[t] for t in themes])
    valid = np.all(np.isfinite(P), axis=1)
    residuals = P - dataset.y[:, None]

    pred_corr = pd.DataFrame(
        np.corrcoef(P[valid].T), index=themes, columns=themes
    ).round(4)
    resid_corr = pd.DataFrame(
        np.corrcoef(residuals[valid].T), index=themes, columns=themes
    ).round(4)

    report["prediction_correlation"] = pred_corr.to_dict()
    report["residual_correlation"] = resid_corr.to_dict()

    offdiag = pred_corr.to_numpy()[np.triu_indices(len(themes), k=1)]
    resid_offdiag = resid_corr.to_numpy()[np.triu_indices(len(themes), k=1)]
    report["mean_prediction_correlation"] = round(float(offdiag.mean()), 4)
    report["mean_residual_correlation"] = round(float(resid_offdiag.mean()), 4)
    logger.info(
        "Mean off-diagonal correlation: predictions %.3f, residuals %.3f",
        offdiag.mean(),
        resid_offdiag.mean(),
    )
    logger.info("Prediction correlation matrix:\n%s", pred_corr.to_string())

    # ------------------------------------------------------------------
    # 3. ablation over combination rules
    # ------------------------------------------------------------------
    logger.info("--- ablation: combination rules ---")
    weights = np.array(
        [
            individual[t]["summary"].get("average_precision_mean", 0.0)
            for t in themes
        ],
        dtype="float64",
    )

    candidates: dict[str, np.ndarray] = {
        "full_model_alone": oof[BENCHMARK_NAME],
    }
    for rule in COMBINERS:
        candidates[rule] = combine(rule, P, weights)

    candidates["thematic_meta_logistic"] = _meta_learner_cv(
        np.nan_to_num(P, nan=0.0), dataset.y, dataset.coords, block_size_m, n_splits, seed
    )
    candidates["full_plus_thematic_mean"] = 0.5 * (
        oof[BENCHMARK_NAME] + combine("thematic_mean", P, weights)
    )

    ablation: dict[str, dict] = {}
    for name, scores in candidates.items():
        ok = np.isfinite(scores) & np.isfinite(dataset.y)
        metrics = compute_metrics(dataset.y[ok], scores[ok], threshold=0.5)
        ablation[name] = {
            **metrics,
            "produces_confidence": name != "full_model_alone",
        }
        logger.info(
            "  %-26s AUC-PR %.4f  ROC-AUC %.4f  confidence=%s",
            name,
            metrics["average_precision"],
            metrics["roc_auc"],
            "yes" if name != "full_model_alone" else "NO",
        )
    report["ablation"] = ablation

    best_overall = max(
        ablation, key=lambda k: ablation[k]["average_precision"]
    )
    best_with_confidence = max(
        (k for k in ablation if ablation[k]["produces_confidence"]),
        key=lambda k: ablation[k]["average_precision"],
    )
    report["best_overall"] = best_overall
    report["best_producing_confidence"] = best_with_confidence
    report["confidence_cost_auc_pr"] = round(
        ablation[best_overall]["average_precision"]
        - ablation[best_with_confidence]["average_precision"],
        4,
    )
    logger.info(
        "Best overall: %r (AUC-PR %.4f). Best that also yields confidence: %r "
        "(AUC-PR %.4f). Cost of keeping the confidence layer: %.4f AUC-PR.",
        best_overall,
        ablation[best_overall]["average_precision"],
        best_with_confidence,
        ablation[best_with_confidence]["average_precision"],
        report["confidence_cost_auc_pr"],
    )

    # ------------------------------------------------------------------
    # 4. per-theme learner comparison
    # ------------------------------------------------------------------
    # Tests whether letting each theme pick its own algorithm beats holding the
    # algorithm fixed. Reported as a deviation candidate, not adopted by
    # default: mixing learners would mean a disagreement between two themes
    # partly reflects the algorithms rather than the physical evidence, and the
    # confidence raster is the product here.
    logger.info("--- per-theme learner comparison ---")
    learner_comparison: dict[str, dict] = {}
    for name in all_models:
        learner_comparison[name] = {}
        for learner in ("catboost", "xgboost", "logreg"):
            result = cross_validate_oof(
                dataset.theme_X(name),
                dataset.y,
                dataset.coords,
                scheme="spatial",
                n_splits=n_splits,
                block_size_m=block_size_m,
                kind=learner,
                imbalance=imbalance,
                categorical_features=dataset.theme_categoricals(name),
                seed=seed,
                label=f"{name} / {learner}",
            )
            learner_comparison[name][learner] = result.summary
        best = max(
            learner_comparison[name],
            key=lambda k: learner_comparison[name][k].get(
                "average_precision_mean", -np.inf
            ),
        )
        learner_comparison[name]["best"] = best
    report["per_theme_learner_comparison"] = learner_comparison
    report["uniform_learner_is_best_everywhere"] = all(
        learner_comparison[n]["best"] == kind for n in all_models
    )

    # ------------------------------------------------------------------
    # 4b. does a MIXED-learner ensemble actually beat the uniform one?
    # ------------------------------------------------------------------
    # Per-theme learner selection can raise an individual theme's score without
    # improving the ensemble, because what the ensemble needs is decorrelated
    # errors rather than strong members. This measures the ensemble-level
    # effect directly, which is the only thing that decides the question.
    #
    # Note the cost side, which no metric here captures: with mixed learners, a
    # disagreement between two themes partly reflects the algorithms rather
    # than the physical evidence, and the confidence raster is the product.
    logger.info("--- mixed-learner ensemble vs uniform-learner ensemble ---")
    mixed_oof: dict[str, np.ndarray] = {}
    mixed_choice: dict[str, str] = {}
    for theme in themes:
        best_learner = learner_comparison[theme]["best"]
        mixed_choice[theme] = best_learner
        if best_learner == kind:
            mixed_oof[theme] = oof[theme]
            continue
        result = cross_validate_oof(
            dataset.theme_X(theme),
            dataset.y,
            dataset.coords,
            scheme="spatial",
            n_splits=n_splits,
            block_size_m=block_size_m,
            kind=best_learner,
            imbalance=imbalance,
            categorical_features=dataset.theme_categoricals(theme),
            seed=seed,
            label=f"{theme} / mixed({best_learner})",
        )
        mixed_oof[theme] = result.oof

    P_mixed = np.column_stack([mixed_oof[t] for t in themes])
    valid_mixed = np.all(np.isfinite(P_mixed), axis=1)
    mixed_corr = np.corrcoef(P_mixed[valid_mixed].T)
    mixed_offdiag = mixed_corr[np.triu_indices(len(themes), k=1)]

    mixed_weights = np.array(
        [
            learner_comparison[t][mixed_choice[t]].get("average_precision_mean", 0.0)
            for t in themes
        ],
        dtype="float64",
    )
    mixed_results = {}
    for rule in ("thematic_mean", "thematic_weighted_mean"):
        scores = combine(rule, P_mixed, mixed_weights)
        ok_m = np.isfinite(scores)
        mixed_results[rule] = compute_metrics(dataset.y[ok_m], scores[ok_m], threshold=0.5)
    blended = 0.5 * (oof[BENCHMARK_NAME] + combine("thematic_mean", P_mixed, mixed_weights))
    ok_m = np.isfinite(blended)
    mixed_results["full_plus_thematic_mean"] = compute_metrics(
        dataset.y[ok_m], blended[ok_m], threshold=0.5
    )

    report["mixed_learner_ensemble"] = {
        "learner_per_theme": mixed_choice,
        "mean_prediction_correlation": round(float(mixed_offdiag.mean()), 4),
        "results": mixed_results,
        "uniform_comparison": {
            rule: ablation[rule]["average_precision"]
            for rule in mixed_results
            if rule in ablation
        },
    }
    for rule, metrics in mixed_results.items():
        uniform_score = ablation.get(rule, {}).get("average_precision")
        logger.info(
            "  %-26s mixed AUC-PR %.4f vs uniform %.4f (delta %+.4f)",
            rule,
            metrics["average_precision"],
            uniform_score if uniform_score is not None else float("nan"),
            (metrics["average_precision"] - uniform_score)
            if uniform_score is not None
            else float("nan"),
        )
    logger.info(
        "Mean prediction correlation: uniform %.3f, mixed %.3f",
        offdiag.mean(),
        mixed_offdiag.mean(),
    )

    # ------------------------------------------------------------------
    # 5. fit final models on all data and persist them
    # ------------------------------------------------------------------
    logger.info("--- fitting final models on the full dataset ---")
    for name in all_models:
        X_model = dataset.theme_X(name)
        cats = dataset.theme_categoricals(name)
        estimator = make_estimator(
            kind=kind,
            seed=seed,
            imbalance=imbalance,
            categorical_features=cats,
            n_positive=int(dataset.y.sum()),
            n_negative=int(len(dataset.y) - dataset.y.sum()),
        )
        estimator.fit(prepare_features(X_model, kind, cats), dataset.y)

        bundle = ModelBundle(
            name=name,
            kind=kind,
            estimator=estimator,
            encoder=dataset.encoder,
            features=list(dataset.theme_features[name]),
            categorical_features=cats,
            threshold=None,  # derived in Stage 5; never defaulted to 0.5
            metadata={
                "seed": seed,
                "imbalance_strategy": imbalance,
                "cv_scheme": "spatial_block",
                "block_size_m": block_size_m,
                "n_splits": n_splits,
                "n_train": int(len(dataset.y)),
                "n_positive": int(dataset.y.sum()),
                "spatial_cv_summary": individual[name]["summary"],
                "is_benchmark": name == BENCHMARK_NAME,
            },
        )
        save_bundle(bundle, cfg.models_dir)

    # ------------------------------------------------------------------
    # 6. persist OOF predictions for stages 5 and 6
    # ------------------------------------------------------------------
    oof_frame = pd.DataFrame({name: oof[name] for name in all_models})
    oof_frame["label"] = dataset.y
    oof_frame["X_coord"] = dataset.coords[:, 0]
    oof_frame["Y_coord"] = dataset.coords[:, 1]
    oof_path = cfg.reports_dir / OOF_FILENAME
    oof_path.parent.mkdir(parents=True, exist_ok=True)
    oof_frame.to_csv(oof_path, index=False)
    logger.info("Out-of-fold predictions written: %s", oof_path)

    out_path = cfg.reports_dir / "models_ablation.json"
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)
    logger.info("Ablation report written: %s", out_path)

    return report
