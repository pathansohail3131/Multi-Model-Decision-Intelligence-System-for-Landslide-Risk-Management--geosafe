"""Stage 6: SHAP interpretability and cross-theme agreement.

Two questions, and the second is the one specific to this architecture.

**What drives each model?** Global SHAP importance per thematic model and for
the full-feature benchmark.

**Where do a factor's thematic and full-model readings agree?** Because the
themes are disjoint, no factor appears in two thematic models, so "agreement
between thematic models on a factor" is not directly measurable. What *is*
measurable, and more useful, is whether a factor means the same thing to its
own thematic model as it does to the full-feature model. A factor whose SHAP
values agree across the two is carrying a self-contained physical signal. A
factor that matters far more in the full model is being used in *interaction*
with factors from other themes — which is precisely the information the
thematic split gives up, and therefore the honest accounting of what that
split costs.

One finding is watched closely throughout. The previous phase found
``dist_road`` to be the dominant feature, and Stage 4 here found the
anthropogenic theme (``dist_road``, ``LULC``, ``NDVI``) to be by far the
strongest thematic model. That is partly genuine road-cut destabilisation and
partly inventory accessibility bias — field-validated databases under-record
landslides in roadless terrain. The two cannot be separated with the available
data, so every statement about ``dist_road`` carries the caveat.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.features.build import prepare_dataset
from src.features.themes import BENCHMARK_NAME
from src.models.bundle import load_bundle
from src.models.estimators import prepare_features
from src.utils.logging_setup import get_logger

__all__ = ["run_shap_stage", "shap_values_for_bundle"]

logger = get_logger(__name__)


def shap_values_for_bundle(bundle, X: pd.DataFrame, y: np.ndarray) -> np.ndarray:
    """Compute per-sample SHAP values for a fitted bundle.

    Parameters
    ----------
    bundle : ModelBundle
        The fitted model.
    X : pandas.DataFrame
        Encoded features restricted to the bundle's feature list.
    y : numpy.ndarray
        Binary target, needed only to construct a CatBoost ``Pool``.

    Returns
    -------
    numpy.ndarray, shape (n_samples, n_features)
        SHAP contribution of each feature to each prediction, in log-odds.

    Notes
    -----
    CatBoost's own ``get_feature_importance(type="ShapValues")`` is used rather
    than the ``shap`` package's ``TreeExplainer``. CatBoost's categorical
    handling uses ordered target statistics that the generic tree explainer
    does not model, so routing through the native implementation is the only
    way to get attributions that correspond to what the model actually did.
    """
    prepared = prepare_features(X, bundle.kind, bundle.categorical_features)

    if bundle.kind == "catboost":
        from catboost import Pool

        cats = [c for c in bundle.categorical_features if c in prepared.columns]
        pool = Pool(prepared, y, cat_features=cats)
        values = bundle.estimator.get_feature_importance(pool, type="ShapValues")
        # Final column is the expected value (base), not a feature.
        return np.asarray(values)[:, :-1]

    import shap

    if bundle.kind == "xgboost":
        explainer = shap.TreeExplainer(bundle.estimator)
        return np.asarray(explainer.shap_values(prepared))

    # Linear pipeline: explain the final estimator on the scaled matrix.
    scaled = bundle.estimator.named_steps["scale"].transform(prepared)
    explainer = shap.LinearExplainer(
        bundle.estimator.named_steps["clf"], scaled
    )
    return np.asarray(explainer.shap_values(scaled))


def _global_importance(shap_values: np.ndarray, features: list[str]) -> dict:
    """Mean absolute SHAP per feature, with signed mean for direction."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    signed = shap_values.mean(axis=0)
    order = np.argsort(-mean_abs)
    return {
        features[i]: {
            "mean_abs_shap": round(float(mean_abs[i]), 5),
            "mean_signed_shap": round(float(signed[i]), 5),
            "rank": int(rank + 1),
            "share_of_total": round(float(mean_abs[i] / mean_abs.sum()), 4)
            if mean_abs.sum() > 0
            else 0.0,
        }
        for rank, i in enumerate(order)
    }


def run_shap_stage(cfg) -> dict:
    """Stage 6 entry point.

    Parameters
    ----------
    cfg : Config
        Resolved project configuration.

    Returns
    -------
    dict
        Report written to ``outputs/reports/interpretability.json``.
    """
    dataset = prepare_dataset(cfg)
    themes = list(cfg.theme_names)
    all_models = themes + [BENCHMARK_NAME]

    report: dict = {"models": {}}
    shap_by_model: dict[str, pd.DataFrame] = {}

    logger.info("--- SHAP global importance per model ---")
    for name in all_models:
        path = cfg.models_dir / f"{name}.joblib"
        if not path.exists():
            logger.warning("bundle %s not found; skipping", path)
            continue
        bundle = load_bundle(path)
        X = dataset.X[bundle.features]
        values = shap_values_for_bundle(bundle, X, dataset.y)
        shap_by_model[name] = pd.DataFrame(values, columns=bundle.features)

        importance = _global_importance(values, bundle.features)
        report["models"][name] = {
            "n_features": len(bundle.features),
            "importance": importance,
        }
        top = list(importance)[: min(5, len(importance))]
        logger.info(
            "  %-16s top features: %s",
            name,
            ", ".join(f"{f} ({importance[f]['share_of_total']:.0%})" for f in top),
        )

    # ------------------------------------------------------------------
    # cross-theme agreement: thematic reading vs full-model reading
    # ------------------------------------------------------------------
    logger.info("--- factor agreement: thematic model vs full model ---")
    agreement: dict[str, dict] = {}
    if BENCHMARK_NAME in shap_by_model:
        full_shap = shap_by_model[BENCHMARK_NAME]
        for theme in themes:
            if theme not in shap_by_model:
                continue
            theme_shap = shap_by_model[theme]
            for feature in theme_shap.columns:
                if feature not in full_shap.columns:
                    continue
                a = theme_shap[feature].to_numpy()
                b = full_shap[feature].to_numpy()
                # Correlation of per-sample attributions: does the feature push
                # the same samples in the same direction in both models?
                corr = (
                    float(np.corrcoef(a, b)[0, 1])
                    if a.std() > 0 and b.std() > 0
                    else float("nan")
                )
                share_theme = float(
                    np.abs(a).mean() / np.abs(theme_shap.to_numpy()).mean(axis=0).sum()
                )
                share_full = float(
                    np.abs(b).mean() / np.abs(full_shap.to_numpy()).mean(axis=0).sum()
                )
                agreement[feature] = {
                    "theme": theme,
                    "shap_correlation_theme_vs_full": round(corr, 4),
                    "importance_share_in_theme": round(share_theme, 4),
                    "importance_share_in_full": round(share_full, 4),
                    "mean_abs_shap_theme": round(float(np.abs(a).mean()), 5),
                    "mean_abs_shap_full": round(float(np.abs(b).mean()), 5),
                }
    report["factor_agreement"] = agreement

    concordant = {
        f: d
        for f, d in agreement.items()
        if np.isfinite(d["shap_correlation_theme_vs_full"])
        and d["shap_correlation_theme_vs_full"] >= 0.5
    }
    report["agreement_summary"] = {
        "n_factors": len(agreement),
        "n_concordant_corr_ge_0.5": len(concordant),
        "concordant_factors": sorted(concordant),
        "mean_correlation": round(
            float(
                np.nanmean(
                    [d["shap_correlation_theme_vs_full"] for d in agreement.values()]
                )
            ),
            4,
        )
        if agreement
        else None,
    }
    logger.info(
        "%d of %d factors read consistently (SHAP corr >= 0.5) between their "
        "thematic model and the full model",
        len(concordant),
        len(agreement),
    )

    # ------------------------------------------------------------------
    # what drives the cases where the themes disagree
    # ------------------------------------------------------------------
    # These are the pixels that will show low confidence on the final map, so
    # knowing which factors are implicated is directly actionable: it tells a
    # survey team what to go and check.
    calibrated_path = cfg.reports_dir / "oof_calibrated.csv"
    if calibrated_path.exists() and shap_by_model:
        calibrated = pd.read_csv(calibrated_path)
        if "disagreement" in calibrated.columns:
            d = calibrated["disagreement"].to_numpy()
            high = d >= np.nanpercentile(d, 90)
            drivers = {}
            for theme in themes:
                if theme not in shap_by_model:
                    continue
                frame = shap_by_model[theme]
                overall = np.abs(frame.to_numpy()).mean(axis=0)
                subset = np.abs(frame.to_numpy()[high]).mean(axis=0)
                drivers[theme] = {
                    feature: {
                        "mean_abs_shap_overall": round(float(overall[i]), 5),
                        "mean_abs_shap_high_disagreement": round(float(subset[i]), 5),
                        "ratio": round(
                            float(subset[i] / overall[i]) if overall[i] > 0 else 0.0, 3
                        ),
                    }
                    for i, feature in enumerate(frame.columns)
                }
            report["high_disagreement_drivers"] = {
                "definition": "top decile of cross-theme disagreement",
                "n_samples": int(high.sum()),
                "by_theme": drivers,
            }
            logger.info(
                "Analysed SHAP drivers for the %d highest-disagreement samples",
                int(high.sum()),
            )

    # ------------------------------------------------------------------
    # the two findings that carry caveats
    # ------------------------------------------------------------------
    watch: dict = {}
    if "anthropogenic" in report["models"]:
        imp = report["models"]["anthropogenic"]["importance"]
        if "dist_road" in imp:
            watch["dist_road"] = {
                **imp["dist_road"],
                "caveat": (
                    "Partly genuine road-cut destabilisation, partly inventory "
                    "accessibility bias: field-validated inventories under-record "
                    "landslides in remote roadless terrain. The two effects cannot "
                    "be separated with the available data."
                ),
            }
    if "geological" in report["models"]:
        imp = report["models"]["geological"]["importance"]
        if "lithology" in imp:
            watch["lithology"] = {
                **imp["lithology"],
                "note": (
                    "Includes the unmapped_coastal_gap level, which marks absent "
                    "GLiM coverage along the coast rather than a rock type. "
                    "Attribution to that level is a coverage indicator."
                ),
            }
    report["flagged_findings"] = watch

    # Persist per-sample SHAP for the full model, for figures and inspection.
    if BENCHMARK_NAME in shap_by_model:
        shap_path = cfg.reports_dir / "shap_full_model.csv"
        shap_by_model[BENCHMARK_NAME].to_csv(shap_path, index=False)
        logger.info("Per-sample SHAP written: %s", shap_path)

    out_path = cfg.reports_dir / "interpretability.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)
    logger.info("Interpretability report written: %s", out_path)

    return report
