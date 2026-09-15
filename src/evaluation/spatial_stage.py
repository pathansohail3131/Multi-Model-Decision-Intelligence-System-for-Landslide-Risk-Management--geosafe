"""Stage 3: spatial versus random CV, and the class-imbalance comparison.

Two questions are settled here, both empirically:

1. **How much optimism does a random split inject?** The full-feature model is
   evaluated under both schemes and the gap is reported. Because the block size
   is itself a judgement call, it is swept rather than fixed, and the gap is
   reported at each size. The shape of that curve is the finding.

2. **Which class-imbalance strategy should the rest of the pipeline use?**
   Four arms are compared under spatial CV, and the winner on AUC-PR is written
   back into the configuration report for Stage 4 to consume.

Neither answer is assumed in advance. The prior expectation — that reweighting
beats SMOTE, because synthetic points interpolated in feature space need not
correspond to real terrain — is recorded in ``docs/methodology.md`` §4.2 so it
can be checked against what actually happened.
"""

from __future__ import annotations

import json

import numpy as np

from src.evaluation.runner import cross_validate_oof
from src.evaluation.spatial_cv import summarise_split
from src.features.build import prepare_dataset
from src.features.themes import BENCHMARK_NAME
from src.models.estimators import IMBALANCE_STRATEGIES
from src.utils.logging_setup import get_logger

__all__ = ["run_spatial_cv_stage", "BLOCK_SIZE_SWEEP"]

logger = get_logger(__name__)

#: Block sizes swept, in metres. Spans from below the measured autocorrelation
#: range to well above it, so the curve shows where the inflation actually
#: comes from rather than asserting one number.
BLOCK_SIZE_SWEEP = (1_000.0, 2_500.0, 5_000.0, 10_000.0, 20_000.0)


def run_spatial_cv_stage(cfg) -> dict:
    """Stage 3 entry point.

    Parameters
    ----------
    cfg : Config
        Resolved project configuration.

    Returns
    -------
    dict
        Report written to ``outputs/reports/spatial_cv.json``.
    """
    dataset = prepare_dataset(cfg)
    seed = cfg.seed
    n_splits = int(cfg.get("cross_validation", "n_splits", default=5))
    kind = str(cfg.get("models", "kind", default="catboost"))

    X_full = dataset.theme_X(BENCHMARK_NAME)
    cats = dataset.theme_categoricals(BENCHMARK_NAME)

    report: dict = {
        "n_samples": int(len(dataset.y)),
        "n_positive": int(dataset.y.sum()),
        "n_splits": n_splits,
        "estimator_kind": kind,
        "seed": seed,
    }

    # ------------------------------------------------------------------
    # 1. how the blocking scheme behaves geometrically, before any model
    # ------------------------------------------------------------------
    logger.info("--- block geometry sweep ---")
    geometry = {}
    for size in BLOCK_SIZE_SWEEP:
        stats = summarise_split(dataset.y, dataset.coords, size, n_splits, seed)
        geometry[f"{int(size)}m"] = stats
        logger.info(
            "  %6.0f m: %5d blocks, %5.1f pts/block, min fold positives %3d, "
            "%.1f%% of test points within 500 m of a training point",
            size,
            stats["n_blocks"],
            stats["mean_points_per_block"],
            stats["min_positives_in_a_fold"],
            100 * stats["test_to_train_distance"]["pct_within_500m"],
        )
    report["block_geometry"] = geometry

    # ------------------------------------------------------------------
    # 2. random CV — the number a naive pipeline would publish
    # ------------------------------------------------------------------
    logger.info("--- random (non-spatial) CV: the inflated baseline ---")
    random_result = cross_validate_oof(
        X_full,
        dataset.y,
        dataset.coords,
        scheme="random",
        n_splits=n_splits,
        kind=kind,
        imbalance="none",
        categorical_features=cats,
        seed=seed,
        label="full / random CV",
    )
    report["random_cv"] = random_result.as_dict()

    # ------------------------------------------------------------------
    # 3. spatial CV at each block size — the honest numbers
    # ------------------------------------------------------------------
    logger.info("--- spatial block CV across block sizes ---")
    sweep = {}
    for size in BLOCK_SIZE_SWEEP:
        result = cross_validate_oof(
            X_full,
            dataset.y,
            dataset.coords,
            scheme="spatial",
            n_splits=n_splits,
            block_size_m=size,
            kind=kind,
            imbalance="none",
            categorical_features=cats,
            seed=seed,
            label=f"full / spatial CV {int(size)}m",
        )
        sweep[f"{int(size)}m"] = result.as_dict()
    report["spatial_cv_sweep"] = sweep

    # Inflation = random minus spatial, per metric. Positive means the random
    # split flattered the model.
    inflation = {}
    for key, result in sweep.items():
        inflation[key] = {
            metric: round(
                random_result.summary.get(f"{metric}_mean", float("nan"))
                - result["summary"].get(f"{metric}_mean", float("nan")),
                4,
            )
            for metric in ("roc_auc", "average_precision")
        }
    report["inflation_vs_random"] = inflation
    logger.info("--- optimism injected by the random split ---")
    for key, values in inflation.items():
        logger.info(
            "  block %-7s ROC-AUC +%.4f, AUC-PR +%.4f (random over spatial)",
            key,
            values["roc_auc"],
            values["average_precision"],
        )

    # ------------------------------------------------------------------
    # 4. choose the block size the rest of the pipeline will use
    # ------------------------------------------------------------------
    configured = cfg.get("cross_validation", "block_size_m", default="auto")
    if configured == "auto":
        # Take the smallest block size beyond which the score has stopped
        # falling materially: that is where blocks are large enough that
        # further separation buys nothing, which is the honest operating
        # point. "Materially" is 0.005 AUC-PR, well inside fold-to-fold noise.
        scores = [
            (size, sweep[f"{int(size)}m"]["summary"].get("average_precision_mean", np.nan))
            for size in BLOCK_SIZE_SWEEP
        ]
        chosen = scores[-1][0]
        for (size, score), (_, next_score) in zip(scores, scores[1:]):
            if abs(score - next_score) < 0.005:
                chosen = size
                break
        block_size = float(chosen)
    else:
        block_size = float(configured)
    report["selected_block_size_m"] = block_size
    logger.info("Selected spatial CV block size: %.0f m", block_size)

    # ------------------------------------------------------------------
    # 5. class-imbalance comparison, under spatial CV at that block size
    # ------------------------------------------------------------------
    logger.info("--- class-imbalance strategy comparison (spatial CV) ---")
    imbalance_results = {}
    for strategy in IMBALANCE_STRATEGIES:
        result = cross_validate_oof(
            X_full,
            dataset.y,
            dataset.coords,
            scheme="spatial",
            n_splits=n_splits,
            block_size_m=block_size,
            kind=kind,
            imbalance=strategy,
            categorical_features=cats,
            seed=seed,
            label=f"full / imbalance={strategy}",
        )
        imbalance_results[strategy] = result.as_dict()
    report["imbalance_comparison"] = imbalance_results

    best = max(
        imbalance_results,
        key=lambda s: imbalance_results[s]["summary"].get(
            "average_precision_mean", -np.inf
        ),
    )
    report["selected_imbalance_strategy"] = best
    logger.info(
        "Selected imbalance strategy: %r (AUC-PR %.4f)",
        best,
        imbalance_results[best]["summary"].get("average_precision_mean", float("nan")),
    )

    out_path = cfg.reports_dir / "spatial_cv.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)
    logger.info("Spatial CV report written: %s", out_path)

    return report
