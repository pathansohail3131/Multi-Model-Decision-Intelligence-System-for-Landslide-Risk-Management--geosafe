"""Stage 5: calibration, the confidence measure, and the operating threshold.

Calibration is not cosmetic here — it is a precondition for the project's
headline output. The confidence raster is derived from disagreement across the
thematic models, and variance over *uncalibrated* scores measures how
differently the models are scaled, not how differently they judge. A
systematically confident model and a systematically conservative one would
appear to disagree everywhere, including on pixels they rank identically. The
confidence layer would then be an artefact of calibration mismatch rather than
a statement about physical evidence.

Calibrators are fitted on out-of-fold predictions, and evaluated through a
second layer of spatial folds, so no calibrator is ever scored on the
predictions it was fitted to.

The operating threshold is derived from an explicit cost ratio and never
defaults to 0.5. 0.5 is the decision boundary of a balanced problem with
symmetric costs; this problem has neither.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

from src.evaluation.metrics import compute_metrics, threshold_candidates
from src.evaluation.spatial_cv import SpatialBlockCV
from src.models.bundle import load_bundle, save_bundle
from src.models.combiners import combine, confidence_from_disagreement, disagreement
from src.models.train import OOF_FILENAME
from src.utils.logging_setup import get_logger

__all__ = ["IsotonicCalibrator", "run_calibration_stage"]

logger = get_logger(__name__)


class IsotonicCalibrator:
    """Monotonic probability calibrator attached to a model bundle.

    Isotonic regression rather than Platt scaling, for two reasons specific to
    this problem: it makes no parametric assumption about the shape of the
    miscalibration, and gradient-boosted trees on an imbalanced target are
    typically miscalibrated in a way that is monotonic but not sigmoid.

    The cost is that isotonic regression can overfit with few samples, which is
    why it is fitted on out-of-fold predictions across the whole dataset rather
    than on a small held-out slice.

    Parameters
    ----------
    y_min, y_max : float
        Output bounds. Clipped away from exactly 0 and 1 so that downstream
        log-odds or noisy-OR operations stay finite.
    """

    def __init__(self, y_min: float = 1e-6, y_max: float = 1 - 1e-6) -> None:
        self.y_min = y_min
        self.y_max = y_max
        self._model = IsotonicRegression(
            y_min=y_min, y_max=y_max, out_of_bounds="clip", increasing=True
        )

    def fit(self, probabilities: np.ndarray, y: np.ndarray) -> "IsotonicCalibrator":
        """Fit the calibrator on out-of-fold probabilities.

        Parameters
        ----------
        probabilities : numpy.ndarray
            Uncalibrated probabilities.
        y : numpy.ndarray
            Binary ground truth.

        Returns
        -------
        IsotonicCalibrator
            The fitted calibrator.
        """
        ok = np.isfinite(probabilities)
        self._model.fit(probabilities[ok], np.asarray(y)[ok])
        return self

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        """Apply calibration.

        Parameters
        ----------
        probabilities : numpy.ndarray
            Uncalibrated probabilities.

        Returns
        -------
        numpy.ndarray
            Calibrated probabilities, finite and strictly inside (0, 1).
        """
        values = np.asarray(probabilities, dtype="float64")
        out = np.full_like(values, np.nan)
        ok = np.isfinite(values)
        out[ok] = self._model.predict(values[ok])
        return out


def _calibrated_oof(
    probabilities: np.ndarray,
    y: np.ndarray,
    coords: np.ndarray,
    block_size_m: float,
    n_splits: int,
    seed: int,
) -> np.ndarray:
    """Produce honestly-evaluated calibrated probabilities.

    A second layer of spatial folds: the calibrator is fitted on the training
    blocks' out-of-fold predictions and applied to the held-out blocks. Fitting
    one calibrator on everything and applying it to everything would report a
    calibration quality that could not be reproduced on new terrain.

    Parameters
    ----------
    probabilities : numpy.ndarray
        Uncalibrated out-of-fold probabilities.
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
        Calibrated out-of-fold probabilities.
    """
    out = np.full(len(y), np.nan, dtype="float64")
    cv = SpatialBlockCV(n_splits=n_splits, block_size_m=block_size_m, random_state=seed)
    for train_idx, test_idx in cv.split(None, y, coords):
        usable = np.isfinite(probabilities[train_idx])
        if usable.sum() < 10 or len(np.unique(y[train_idx][usable])) < 2:
            out[test_idx] = probabilities[test_idx]
            continue
        calibrator = IsotonicCalibrator().fit(
            probabilities[train_idx], y[train_idx]
        )
        out[test_idx] = calibrator.transform(probabilities[test_idx])
    return out


def run_calibration_stage(cfg) -> dict:
    """Stage 5 entry point.

    Parameters
    ----------
    cfg : Config
        Resolved project configuration.

    Returns
    -------
    dict
        Report written to ``outputs/reports/calibration_threshold.json``.
    """
    oof_path = cfg.reports_dir / OOF_FILENAME
    if not oof_path.exists():
        raise FileNotFoundError(
            f"out-of-fold predictions not found at {oof_path}; run stage 4 first"
        )
    frame = pd.read_csv(oof_path)

    ablation_path = cfg.reports_dir / "models_ablation.json"
    with ablation_path.open("r", encoding="utf-8") as handle:
        ablation_report = json.load(handle)

    block_size_m = float(ablation_report["block_size_m"])
    n_splits = int(ablation_report["n_splits"])
    seed = cfg.seed
    themes = list(ablation_report["themes"])

    y = frame["label"].to_numpy().astype(int)
    coords = frame[["X_coord", "Y_coord"]].to_numpy(dtype="float64")

    report: dict = {"themes": themes, "block_size_m": block_size_m, "seed": seed}

    # ------------------------------------------------------------------
    # 1. calibrate each thematic model
    # ------------------------------------------------------------------
    logger.info("--- calibrating thematic models ---")
    raw_P, cal_P = [], []
    calibration_quality = {}

    for theme in themes:
        raw = frame[theme].to_numpy(dtype="float64")
        calibrated = _calibrated_oof(raw, y, coords, block_size_m, n_splits, seed)

        ok = np.isfinite(raw) & np.isfinite(calibrated)
        before = float(brier_score_loss(y[ok], raw[ok]))
        after = float(brier_score_loss(y[ok], calibrated[ok]))
        calibration_quality[theme] = {
            "brier_before": round(before, 5),
            "brier_after": round(after, 5),
            "improvement": round(before - after, 5),
            "mean_prob_before": round(float(np.nanmean(raw)), 5),
            "mean_prob_after": round(float(np.nanmean(calibrated)), 5),
            "observed_positive_rate": round(float(y.mean()), 5),
        }
        logger.info(
            "  %-16s Brier %.5f -> %.5f | mean p %.4f -> %.4f (base rate %.4f)",
            theme,
            before,
            after,
            np.nanmean(raw),
            np.nanmean(calibrated),
            y.mean(),
        )
        raw_P.append(raw)
        cal_P.append(calibrated)

    raw_P = np.column_stack(raw_P)
    cal_P = np.column_stack(cal_P)
    report["calibration"] = calibration_quality

    # ------------------------------------------------------------------
    # 2. why calibration had to come first
    # ------------------------------------------------------------------
    # Quantify how much the disagreement measure changes once the models are
    # on a common scale. A large change means the uncalibrated confidence
    # raster would have been substantially an artefact.
    d_raw = disagreement(raw_P, "std")
    d_cal = disagreement(cal_P, "std")
    ok = np.isfinite(d_raw) & np.isfinite(d_cal)
    report["disagreement_shift"] = {
        "mean_disagreement_uncalibrated": round(float(np.nanmean(d_raw)), 5),
        "mean_disagreement_calibrated": round(float(np.nanmean(d_cal)), 5),
        "correlation_between_them": round(
            float(np.corrcoef(d_raw[ok], d_cal[ok])[0, 1]), 4
        ),
    }
    logger.info(
        "Disagreement before/after calibration: %.4f -> %.4f (corr %.3f)",
        np.nanmean(d_raw),
        np.nanmean(d_cal),
        report["disagreement_shift"]["correlation_between_them"],
    )

    # ------------------------------------------------------------------
    # 3. build the production susceptibility score
    # ------------------------------------------------------------------
    weights = np.array(
        [
            ablation_report["individual_models"][t]["summary"].get(
                "average_precision_mean", 0.0
            )
            for t in themes
        ],
        dtype="float64",
    )
    full_raw = frame["full"].to_numpy(dtype="float64")
    full_cal = _calibrated_oof(full_raw, y, coords, block_size_m, n_splits, seed)

    thematic_mean = combine("thematic_mean", cal_P, weights)
    susceptibility = 0.5 * (full_cal + thematic_mean)

    report["production_rule"] = "full_plus_thematic_mean (calibrated)"
    ok = np.isfinite(susceptibility)
    report["production_score_metrics"] = compute_metrics(
        y[ok], susceptibility[ok], threshold=0.5
    )

    # ------------------------------------------------------------------
    # 4. confidence from disagreement
    # ------------------------------------------------------------------
    confidence = confidence_from_disagreement(d_cal)
    report["confidence_summary"] = {
        "mean": round(float(np.nanmean(confidence)), 4),
        "median": round(float(np.nanmedian(confidence)), 4),
        "p10": round(float(np.nanpercentile(confidence, 10)), 4),
        "p90": round(float(np.nanpercentile(confidence, 90)), 4),
        "method": "1 - std(calibrated thematic probabilities) / max observed",
    }

    # ------------------------------------------------------------------
    # 5. threshold selection
    # ------------------------------------------------------------------
    logger.info("--- operating threshold selection ---")
    cost_fn = float(cfg.get("threshold", "cost_false_negative", default=10))
    cost_fp = float(cfg.get("threshold", "cost_false_positive", default=1))
    floor = float(cfg.get("threshold", "precision_floor", default=0.25))

    candidates = threshold_candidates(
        y[ok], susceptibility[ok], cost_fn=cost_fn, cost_fp=cost_fp, precision_floor=floor
    )
    table = {}
    for name, threshold in candidates.items():
        metrics = compute_metrics(y[ok], susceptibility[ok], threshold=threshold)
        table[name] = metrics
        logger.info(
            "  %-18s t=%.4f  recall %.3f  precision %.3f  F2 %.3f  kappa %.3f  "
            "flags %.1f%% of samples",
            name,
            threshold,
            metrics["recall"],
            metrics["precision"],
            metrics["f2"],
            metrics["cohen_kappa"],
            100 * metrics["flagged_rate"],
        )
    report["threshold_candidates"] = table

    selected_name = str(cfg.get("threshold", "method", default="cost_minimising"))
    selected = float(candidates.get(selected_name, candidates["cost_minimising"]))
    report["selected_threshold"] = selected
    report["selected_threshold_method"] = selected_name
    logger.info("Selected threshold: %.4f via %r", selected, selected_name)

    # Sensitivity of the threshold to the assumed cost ratio. The 10:1 figure
    # is a stated assumption, not a measured quantity, so how much the answer
    # moves with it is part of the result.
    sensitivity = {}
    for ratio in (2, 5, 10, 20, 50):
        threshold, _ = __import__(
            "src.evaluation.metrics", fromlist=["cost_minimising_threshold"]
        ).cost_minimising_threshold(y[ok], susceptibility[ok], float(ratio), 1.0)
        metrics = compute_metrics(y[ok], susceptibility[ok], threshold=threshold)
        sensitivity[f"{ratio}:1"] = {
            "threshold": round(threshold, 4),
            "recall": round(metrics["recall"], 4),
            "precision": round(metrics["precision"], 4),
            "flagged_rate": round(metrics["flagged_rate"], 4),
        }
    report["cost_ratio_sensitivity"] = sensitivity
    logger.info("Cost-ratio sensitivity: %s", json.dumps(sensitivity))

    # ------------------------------------------------------------------
    # 6. the four-quadrant decision table — the actual product
    # ------------------------------------------------------------------
    conf_median = float(np.nanmedian(confidence))
    high_susc = susceptibility >= selected
    high_conf = confidence >= conf_median
    quadrants = {
        "act_high_susc_high_conf": int((high_susc & high_conf).sum()),
        "survey_high_susc_low_conf": int((high_susc & ~high_conf).sum()),
        "monitor_low_susc_low_conf": int((~high_susc & ~high_conf).sum()),
        "deprioritise_low_susc_high_conf": int((~high_susc & high_conf).sum()),
    }
    positives_by_quadrant = {
        "act_high_susc_high_conf": int(y[high_susc & high_conf].sum()),
        "survey_high_susc_low_conf": int(y[high_susc & ~high_conf].sum()),
        "monitor_low_susc_low_conf": int(y[~high_susc & ~high_conf].sum()),
        "deprioritise_low_susc_high_conf": int(y[~high_susc & high_conf].sum()),
    }
    report["decision_quadrants"] = {
        "confidence_split_at": round(conf_median, 4),
        "counts": quadrants,
        "known_landslides": positives_by_quadrant,
    }
    logger.info("Decision quadrants (samples / known landslides):")
    for key in quadrants:
        logger.info(
            "  %-34s %5d / %3d", key, quadrants[key], positives_by_quadrant[key]
        )

    # ------------------------------------------------------------------
    # 7. attach calibrators and the threshold to the saved bundles
    # ------------------------------------------------------------------
    logger.info("--- attaching calibrators and threshold to model bundles ---")
    for name in themes + ["full"]:
        path = cfg.models_dir / f"{name}.joblib"
        if not path.exists():
            logger.warning("bundle %s not found; skipping", path)
            continue
        bundle = load_bundle(path)
        raw = frame[name].to_numpy(dtype="float64")
        bundle.calibrator = IsotonicCalibrator().fit(raw, y)
        bundle.threshold = selected
        bundle.metadata["calibration"] = calibration_quality.get(name, {})
        bundle.metadata["threshold_method"] = selected_name
        save_bundle(bundle, cfg.models_dir)

    # Persist the per-sample outputs so Stage 6 and the docs can reuse them.
    frame_out = frame.copy()
    for i, theme in enumerate(themes):
        frame_out[f"{theme}_calibrated"] = cal_P[:, i]
    frame_out["full_calibrated"] = full_cal
    frame_out["susceptibility"] = susceptibility
    frame_out["disagreement"] = d_cal
    frame_out["confidence"] = confidence
    frame_out.to_csv(cfg.reports_dir / "oof_calibrated.csv", index=False)

    out_path = cfg.reports_dir / "calibration_threshold.json"
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)
    logger.info("Calibration and threshold report written: %s", out_path)

    return report
