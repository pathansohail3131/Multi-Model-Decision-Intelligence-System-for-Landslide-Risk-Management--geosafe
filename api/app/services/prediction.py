"""Prediction service: the trained bundles, loaded once and applied per request.

Nothing here is a second implementation of the model. Scoring goes through each
bundle's own encoder, estimator and attached calibrator; the production blend,
disagreement and confidence come from :mod:`src.models.combiners`; the
susceptibility classes from :mod:`src.inference.raster`; and the planner
quadrant wording from :mod:`src.decision.layers`. What this module adds is
loading, request-shaped inputs and outputs, and honest reporting of what each
number is and where its reference values came from.
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from api.app.core.errors import ServiceError, ServiceNotReady
from api.app.schemas.factors import CATEGORICAL, CATEGORY_LEVELS, FACTORS, SHAP_SOURCE_FACTOR
from api.app.services.places import get_place_index
from src.data.loaders import load_training_data
from src.decision.layers import QUADRANT_INTERPRETATION
from src.features.themes import BENCHMARK_NAME
from src.inference.raster import susceptibility_classes
from src.models.bundle import load_bundle
from src.models.combiners import (
    confidence_from_disagreement,
    disagreement,
    full_plus_thematic_mean,
)
from src.models.estimators import prepare_features
from src.utils.logging_setup import get_logger

__all__ = ["PredictionService"]

logger = get_logger(__name__)

THEME_LABELS = {
    "topographic": "Topographic",
    "hydrological": "Hydrological",
    "geological": "Geological",
    "anthropogenic": "Anthropogenic",
    BENCHMARK_NAME: "Full-feature",
}

CLASS_LABELS = {
    "very_low": "Very low",
    "low": "Low",
    "moderate": "Moderate",
    "high": "High",
    "very_high": "Very high",
}

DECISION_LABELS = {
    "act_now": "Act now",
    "survey_first": "Survey first",
    "monitor": "Monitor",
    "deprioritise": "Deprioritise",
}

#: Categorical levels with fewer training samples than this are flagged.
SPARSE_LEVEL_SAMPLES = 50

CAVEATS = [
    "dist_road is the strongest single factor. That is partly genuine road-cut "
    "destabilisation and partly inventory accessibility bias: field inventories "
    "under-record landslides in roadless terrain. The two cannot be separated "
    "with the available data.",
    "dist_fault is distance to DEM-derived structural lineaments, a proxy — not "
    "mapped geological faults.",
    "Rainfall is a CHIRPS 5-year annual mean. It carries no storm timing, so "
    "these predictions describe predisposition, not a forecast of when a "
    "slope will fail.",
    "Confidence is relative to this dataset: it ranks how strongly the four "
    "thematic models agree. It is not a calibrated probability of being "
    "correct.",
]


def _round(value, digits: int = 6):
    if value is None:
        return None
    value = float(value)
    return round(value, digits) if np.isfinite(value) else None


class PredictionService:
    """Load the thematic and full-feature bundles and score scenarios with them.

    Loading runs on a background thread because unpickling five CatBoost
    bundles and importing the ML stack takes tens of seconds; the health
    endpoint reports genuine progress in the meantime rather than blocking.

    Parameters
    ----------
    cfg : Config
        Resolved project configuration.
    """

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.status = "idle"
        self.stage = "Waiting to start"
        self.error: str | None = None
        self.started_at: float | None = None
        self.ready_at: float | None = None
        self._predict_lock = threading.Lock()
        self._state: dict | None = None

    # ------------------------------------------------------------------
    # loading
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Begin loading on a daemon thread."""
        self.started_at = time.perf_counter()
        self.status = "loading"
        threading.Thread(target=self._load, name="model-loader", daemon=True).start()

    def _load(self) -> None:
        try:
            self._state = self._build_state()
            self.status = "ready"
            self.stage = "Ready"
            self.ready_at = time.perf_counter()
            logger.info(
                "Prediction service ready in %.1f s", self.ready_at - (self.started_at or 0)
            )
        except Exception as exc:  # noqa: BLE001 - surfaced through /api/health
            logger.exception("Prediction service failed to load")
            self.status = "error"
            self.error = str(exc)
            self.stage = "Failed"

    def _build_state(self) -> dict:
        cfg = self.cfg
        themes = list(cfg.theme_names)
        names = themes + [BENCHMARK_NAME]

        self.stage = "Loading model bundles"
        bundles = {}
        for name in names:
            path = cfg.models_dir / f"{name}.joblib"
            if not path.exists():
                raise ServiceError(
                    f"Model bundle {path.name} was not found in {cfg.models_dir}. "
                    "Train the models first: python run_pipeline.py stage 1 --to 5"
                )
            bundles[name] = load_bundle(path)

        incomplete = [n for n, b in bundles.items() if b.threshold is None or b.calibrator is None]
        if incomplete:
            raise ServiceError(
                f"Bundles {incomplete} have no calibrator or operating threshold attached. "
                "Stage 4 has run but Stage 5 has not: python run_pipeline.py stage 5"
            )
        thresholds = {round(float(b.threshold), 10) for b in bundles.values()}
        if len(thresholds) != 1:
            raise ServiceError(
                "The model bundles carry different operating thresholds, so they come "
                "from different pipeline runs. Re-run: python run_pipeline.py stage 4 --to 5"
            )
        threshold = float(bundles[BENCHMARK_NAME].threshold)

        self.stage = "Reading training data"
        training, _ = load_training_data(cfg)

        self.stage = "Reading pipeline reports"
        reports = self._read_reports(threshold)

        return {
            "themes": themes,
            "bundles": bundles,
            "threshold": threshold,
            "training": training,
            "stats": self._training_stats(training),
            "reports": reports,
            "loaded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }

    def _read_reports(self, threshold: float) -> dict:
        """Read the Stage 5 reference values that give confidence its scale."""
        reports_dir = self.cfg.reports_dir
        out: dict = {"calibration": None, "confidence": None, "warnings": []}

        calibration_path = reports_dir / "calibration_threshold.json"
        if calibration_path.exists():
            report = json.loads(calibration_path.read_text(encoding="utf-8"))
            selected = report.get("selected_threshold")
            if selected is not None and abs(float(selected) - threshold) > 1e-9:
                out["warnings"].append(
                    "calibration_threshold.json does not match the loaded bundles' threshold; "
                    "the reports and models come from different runs."
                )
            else:
                out["calibration"] = report
        else:
            out["warnings"].append("calibration_threshold.json not found; run Stage 5.")

        oof_path = reports_dir / "oof_calibrated.csv"
        split = None
        if out["calibration"]:
            split = out["calibration"].get("decision_quadrants", {}).get("confidence_split_at")
        if oof_path.exists() and split is not None:
            oof = pd.read_csv(oof_path, usecols=["disagreement"])
            ceiling = float(np.nanmax(oof["disagreement"].to_numpy()))
            out["confidence"] = {
                "ceiling": ceiling,
                "split": float(split),
                "n_reference": int(oof["disagreement"].notna().sum()),
            }
        else:
            out["warnings"].append(
                "oof_calibrated.csv or the confidence split is missing, so confidence and the "
                "planner decision cannot be scaled. Run Stage 5."
            )
        for message in out["warnings"]:
            logger.warning(message)
        return out

    @staticmethod
    def _training_stats(df: pd.DataFrame) -> dict:
        stats: dict = {}
        positive = df["label"] == 1
        for factor in FACTORS:
            values = df[factor].astype("float64")
            if factor in CATEGORICAL:
                levels = []
                for code, (key, label) in CATEGORY_LEVELS[factor].items():
                    mask = values.round().astype(int) == code
                    n = int(mask.sum())
                    if n == 0:
                        continue
                    hits = int((mask & positive).sum())
                    levels.append(
                        {
                            "code": int(code),
                            "key": key,
                            "label": label,
                            "n_samples": n,
                            "n_landslides": hits,
                            "landslide_rate": round(hits / n, 4),
                        }
                    )
                stats[factor] = {"levels": levels}
                continue
            q = np.nanpercentile(values, [0, 1, 5, 25, 50, 75, 95, 99, 100])
            stats[factor] = {
                "min": float(q[0]),
                "p01": float(q[1]),
                "p05": float(q[2]),
                "p25": float(q[3]),
                "median": float(q[4]),
                "p75": float(q[5]),
                "p95": float(q[6]),
                "p99": float(q[7]),
                "max": float(q[8]),
                "median_landslide": float(np.nanmedian(values[positive])),
                "median_non_landslide": float(np.nanmedian(values[~positive])),
            }
        return stats

    # ------------------------------------------------------------------
    # access
    # ------------------------------------------------------------------
    def _require(self) -> dict:
        if self.status != "ready" or self._state is None:
            if self.status == "error":
                raise ServiceError(self.error or "the prediction service failed to load")
            raise ServiceNotReady(f"models are still loading ({self.stage})")
        return self._state

    def health(self) -> dict:
        elapsed = None
        if self.started_at is not None:
            end = self.ready_at or time.perf_counter()
            elapsed = round(end - self.started_at, 1)
        state = self._state
        return {
            "status": self.status,
            "stage": self.stage,
            "error": self.error,
            "elapsed_s": elapsed,
            "models": list(state["bundles"]) if state else [],
            "report_warnings": state["reports"]["warnings"] if state else [],
        }

    # ------------------------------------------------------------------
    # metadata
    # ------------------------------------------------------------------
    def metadata(self) -> dict:
        state = self._require()
        cfg = self.cfg
        bundles = state["bundles"]
        stats = state["stats"]
        calibration = state["reports"]["calibration"]
        confidence = state["reports"]["confidence"]
        threshold = state["threshold"]

        factor_theme = {
            factor: theme
            for theme in state["themes"]
            for factor in cfg.get("themes", theme, "factors")
        }
        factors = []
        for theme in state["themes"]:
            for name in cfg.get("themes", theme, "factors"):
                spec = FACTORS[name]
                entry = {
                    "name": name,
                    "label": spec["label"],
                    "unit": spec["unit"],
                    "theme": factor_theme[name],
                    "description": spec["description"],
                    "kind": "categorical" if name in CATEGORICAL else "numeric",
                }
                if name in CATEGORICAL:
                    entry["levels"] = stats[name]["levels"]
                else:
                    entry.update(
                        bounds=spec["bounds"],
                        decimals=spec["decimals"],
                        scale=spec["scale"],
                        training=stats[name],
                    )
                factors.append(entry)

        def model_card(name: str) -> dict:
            bundle = bundles[name]
            cv = bundle.metadata.get("spatial_cv_summary", {})
            return {
                "name": name,
                "label": THEME_LABELS.get(name, name),
                "role": "benchmark" if name == BENCHMARK_NAME else "thematic",
                "algorithm": bundle.kind,
                "features": list(bundle.features),
                "factors": list(cfg.get("themes", name, "factors"))
                if name != BENCHMARK_NAME
                else list(cfg.raster_factors),
                "calibrated": bundle.calibrator is not None,
                "created_utc": bundle.metadata.get("created_utc"),
                "spatial_cv": {
                    "scheme": bundle.metadata.get("cv_scheme"),
                    "block_size_m": bundle.metadata.get("block_size_m"),
                    "n_splits": bundle.metadata.get("n_splits"),
                    "roc_auc_mean": _round(cv.get("roc_auc_mean"), 4),
                    "roc_auc_std": _round(cv.get("roc_auc_std"), 4),
                    "average_precision_mean": _round(cv.get("average_precision_mean"), 4),
                    "average_precision_std": _round(cv.get("average_precision_std"), 4),
                }
                if cv
                else None,
            }

        threshold_info: dict = {
            "value": threshold,
            "method": bundles[BENCHMARK_NAME].metadata.get("threshold_method"),
            "cost_false_negative": cfg.get("threshold", "cost_false_negative"),
            "cost_false_positive": cfg.get("threshold", "cost_false_positive"),
            "oof_metrics": None,
        }
        production: dict = {
            "rule": "0.5 × (full-feature model + mean of the 4 thematic models), all calibrated",
            "oof_metrics": None,
        }
        if calibration:
            selected = calibration.get("threshold_candidates", {}).get(
                calibration.get("selected_threshold_method", ""), {}
            )
            if selected:
                threshold_info["oof_metrics"] = {
                    key: _round(selected.get(key), 4)
                    for key in ("recall", "precision", "f2", "flagged_rate")
                }
                threshold_info["oof_metrics"]["n"] = selected.get("n")
            metrics = calibration.get("production_score_metrics", {})
            production["oof_metrics"] = {
                "roc_auc": _round(metrics.get("roc_auc"), 4),
                "average_precision": _round(metrics.get("average_precision"), 4),
                "n": metrics.get("n"),
            }

        training = state["training"]
        return {
            "project": {
                "name": cfg.get("project", "name"),
                "region": cfg.get("project", "region"),
            },
            "factors": factors,
            "models": [model_card(name) for name in state["themes"] + [BENCHMARK_NAME]],
            "threshold": threshold_info,
            "production": production,
            "classes": [
                {"key": key, "label": CLASS_LABELS[key], "range": [low, min(high, 1.0)]}
                for key, (low, high) in susceptibility_classes(threshold).items()
            ],
            "decisions": [
                {"key": key, "label": DECISION_LABELS[key], "interpretation": text}
                for key, text in QUADRANT_INTERPRETATION.items()
            ],
            "confidence": {
                "available": confidence is not None,
                "ceiling": _round(confidence["ceiling"]) if confidence else None,
                "split": _round(confidence["split"]) if confidence else None,
                "method": "1 − std(calibrated thematic probabilities) ÷ largest std across "
                "the out-of-fold training predictions (Stage 5)",
                "n_reference": confidence["n_reference"] if confidence else None,
            },
            "dataset": {
                "n_samples": int(len(training)),
                "n_landslides": int(training["label"].sum()),
                "cell_size_m": cfg.get("grid", "resolution_m"),
            },
            "caveats": CAVEATS,
            "report_warnings": state["reports"]["warnings"],
            "loaded_utc": state["loaded_utc"],
        }

    # ------------------------------------------------------------------
    # scoring
    # ------------------------------------------------------------------
    def _frame(self, rows: list[dict]) -> pd.DataFrame:
        return pd.DataFrame(rows, columns=self.cfg.raster_factors).astype("float64")

    def _score(self, state: dict, frame: pd.DataFrame) -> dict:
        """Score a frame with every bundle. Returns raw and calibrated arrays."""
        raw, calibrated = {}, {}
        for name, bundle in state["bundles"].items():
            # Same path as ModelBundle.predict_proba(calibrated=True), split in
            # two so the uncalibrated score can be reported alongside.
            raw[name] = bundle.predict_proba(frame, calibrated=False)
            calibrated[name] = bundle.calibrator.transform(raw[name])

        thematic = np.column_stack([calibrated[t] for t in state["themes"]])
        return {
            "raw": raw,
            "calibrated": calibrated,
            "thematic": thematic,
            "susceptibility": full_plus_thematic_mean(calibrated[BENCHMARK_NAME], thematic),
        }

    @staticmethod
    def _class_for(value: float, threshold: float) -> dict:
        for key, (low, high) in susceptibility_classes(threshold).items():
            if low <= value < high:
                return {"key": key, "label": CLASS_LABELS[key], "range": [low, min(high, 1.0)]}
        return {"key": "very_high", "label": CLASS_LABELS["very_high"], "range": [0.4, 1.0]}

    def _explain(self, bundle, frame: pd.DataFrame) -> dict | None:
        """Per-row SHAP for one bundle, in the raw model's log-odds.

        Uses CatBoost's native ShapValues, as Stage 6 does, because its ordered
        categorical statistics are not modelled by generic tree explainers.
        """
        if bundle.kind != "catboost":
            return None
        from catboost import Pool

        encoded = bundle.encoder.transform(frame)[bundle.features]
        prepared = prepare_features(encoded, bundle.kind, bundle.categorical_features)
        categorical = [c for c in bundle.categorical_features if c in prepared.columns]
        values = np.asarray(
            bundle.estimator.get_feature_importance(
                Pool(prepared, cat_features=categorical), type="ShapValues"
            )
        )[0]
        contributions, base = values[:-1], float(values[-1])

        features = []
        by_factor: dict[str, float] = {}
        for feature, shap_value in zip(bundle.features, contributions):
            encoded_value = encoded.iloc[0][feature]
            factor = SHAP_SOURCE_FACTOR.get(feature, feature)
            features.append(
                {
                    "feature": feature,
                    "factor": factor,
                    "encoded_value": str(encoded_value)
                    if feature in bundle.categorical_features
                    else _round(encoded_value),
                    "shap": _round(shap_value),
                }
            )
            by_factor[factor] = by_factor.get(factor, 0.0) + float(shap_value)

        return {
            "model": bundle.name,
            "base_log_odds": _round(base),
            "output_log_odds": _round(base + float(contributions.sum())),
            "features": features,
            "by_factor": [
                {"factor": f, "shap": _round(v)}
                for f, v in sorted(by_factor.items(), key=lambda item: -abs(item[1]))
            ],
        }

    def _warnings(self, state: dict, values: dict) -> list[dict]:
        stats = state["stats"]
        out = []
        for factor, spec in FACTORS.items():
            value = values[factor]
            if factor in CATEGORICAL:
                level = next(
                    (lv for lv in stats[factor]["levels"] if lv["code"] == int(value)), None
                )
                name = CATEGORY_LEVELS[factor][int(value)][1]
                if level is None:
                    out.append(
                        {
                            "factor": factor,
                            "kind": "unseen_level",
                            "message": f"{spec['label']} '{name}' never occurs in the training data; "
                            "the models have no evidence for it.",
                        }
                    )
                elif level["n_samples"] < SPARSE_LEVEL_SAMPLES:
                    out.append(
                        {
                            "factor": factor,
                            "kind": "sparse_level",
                            "message": f"{spec['label']} '{name}' has only {level['n_samples']} training "
                            f"samples ({level['n_landslides']} landslides); its effect is poorly constrained.",
                        }
                    )
                continue
            low, high = stats[factor]["min"], stats[factor]["max"]
            if value < low or value > high:
                out.append(
                    {
                        "factor": factor,
                        "kind": "outside_training_range",
                        "message": f"{spec['label']} {value:g} is outside the training range "
                        f"{low:g} to {high:g}; the models are extrapolating.",
                    }
                )

        tolerance = float(self.cfg.get("features", "flat_slope_tolerance", default=0.0))
        if values["slope"] <= tolerance:
            out.append(
                {
                    "factor": "aspect",
                    "kind": "flat_cell",
                    "message": "Slope is 0, so the cell is encoded as flat and aspect is ignored.",
                }
            )
        if int(values["lithology"]) == -1:
            out.append(
                {
                    "factor": "lithology",
                    "kind": "coverage_gap",
                    "message": "Lithology 'unmapped' marks missing GLiM coverage along the coast, "
                    "not a rock type.",
                }
            )
        return out

    def predict(self, values: dict, explain: bool = True) -> dict:
        """Score one scenario with the production rule.

        Parameters
        ----------
        values : dict
            The 14 raw conditioning factors.
        explain : bool
            Whether to compute per-row SHAP attributions.
        """
        state = self._require()
        frame = self._frame([values])
        themes = state["themes"]
        threshold = state["threshold"]
        confidence_ref = state["reports"]["confidence"]

        with self._predict_lock:
            started = time.perf_counter()
            scored = self._score(state, frame)
            inference_ms = (time.perf_counter() - started) * 1000

            explanations = None
            explain_ms = None
            if explain:
                started = time.perf_counter()
                explanations = {
                    name: self._explain(bundle, frame)
                    for name, bundle in state["bundles"].items()
                }
                explain_ms = (time.perf_counter() - started) * 1000

        susceptibility = float(scored["susceptibility"][0])
        spread = float(disagreement(scored["thematic"], "std")[0])

        confidence = None
        decision = None
        if confidence_ref is not None:
            confidence = float(
                confidence_from_disagreement(
                    np.array([spread]), max_possible=confidence_ref["ceiling"]
                )[0]
            )
            high_susceptibility = susceptibility >= threshold
            high_confidence = confidence >= confidence_ref["split"]
            if high_susceptibility and high_confidence:
                key = "act_now"
            elif high_susceptibility:
                key = "survey_first"
            elif high_confidence:
                key = "deprioritise"
            else:
                key = "monitor"
            decision = {
                "key": key,
                "label": DECISION_LABELS[key],
                "interpretation": QUADRANT_INTERPRETATION[key],
            }

        models = []
        for name in themes + [BENCHMARK_NAME]:
            models.append(
                {
                    "name": name,
                    "label": THEME_LABELS.get(name, name),
                    "role": "benchmark" if name == BENCHMARK_NAME else "thematic",
                    "probability": _round(scored["calibrated"][name][0]),
                    "raw_probability": _round(scored["raw"][name][0]),
                }
            )

        return {
            "prediction": {
                "susceptibility": _round(susceptibility),
                "threshold": _round(threshold),
                "above_threshold": bool(susceptibility >= threshold),
                "class": self._class_for(susceptibility, threshold),
                "full_model": _round(scored["calibrated"][BENCHMARK_NAME][0]),
                "thematic_mean": _round(scored["thematic"][0].mean()),
                "disagreement": _round(spread),
                "confidence": _round(confidence),
                "confidence_split": _round(confidence_ref["split"]) if confidence_ref else None,
                "decision": decision,
            },
            "models": models,
            "explanations": explanations,
            "warnings": self._warnings(state, values),
            "inputs": values,
            "timing_ms": {
                "inference": round(inference_ms, 1),
                "explanation": round(explain_ms, 1) if explain_ms is not None else None,
            },
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }

    def sensitivity(self, values: dict, factor: str, points: int) -> dict:
        """Re-score the scenario across one factor's range, holding the rest fixed.

        Each point is a real model evaluation — this is an individual
        conditional expectation curve for this scenario, not a global partial
        dependence plot.
        """
        state = self._require()
        stats = state["stats"]

        if factor in CATEGORICAL:
            grid = [float(level["code"]) for level in stats[factor]["levels"]]
            basis = "every level present in the training data"
        else:
            if factor == "aspect":
                grid = list(np.linspace(0.0, 360.0, points))
                basis = "the full compass"
            else:
                # Percentile spacing puts evaluations where the training data
                # actually lies; even spacing would spend most points on the
                # sparse tail of skewed factors such as distances.
                observed = state["training"][factor].astype("float64").to_numpy()
                grid = list(np.unique(np.nanpercentile(observed, np.linspace(1, 99, points))))
                basis = "the 1st to 99th percentile of the training data, spaced by percentile"
                if len(grid) < 5:
                    s = stats[factor]
                    grid = list(np.linspace(s["p01"], s["p99"], points))
            grid = sorted(set(float(g) for g in grid) | {float(values[factor])})

        rows = [{**values, factor: g} for g in grid]
        frame = self._frame(rows)
        with self._predict_lock:
            scored = self._score(state, frame)

        return {
            "factor": factor,
            "values": [_round(g) for g in grid],
            "current_value": _round(values[factor]),
            "susceptibility": [_round(v) for v in scored["susceptibility"]],
            "full_model": [_round(v) for v in scored["calibrated"][BENCHMARK_NAME]],
            "thematic": {
                theme: [_round(v) for v in scored["thematic"][:, i]]
                for i, theme in enumerate(state["themes"])
            },
            "threshold": _round(state["threshold"]),
            "range_basis": basis,
        }

    def training_sample(self, label: int | None = None, sample_id: int | None = None) -> dict:
        """Return one real row of the training data as a scenario."""
        state = self._require()
        df = state["training"]
        if sample_id is not None:
            subset = df[df["sample_id"] == sample_id]
            if subset.empty:
                raise KeyError(f"no training sample with id {sample_id}")
            row = subset.iloc[0]
        else:
            subset = df if label is None else df[df["label"] == label]
            row = subset.iloc[int(np.random.default_rng().integers(len(subset)))]

        factors = {}
        for factor in self.cfg.raster_factors:
            value = float(row[factor])
            factors[factor] = int(round(value)) if factor in CATEGORICAL else value
        taluk = row.get("taluk")
        places = get_place_index()
        return {
            "sample_id": int(row["sample_id"]),
            "label": int(row["label"]),
            "taluk": None if pd.isna(taluk) else str(taluk),
            "x": float(row["X"]),
            "y": float(row["Y"]),
            "factors": factors,
            "note": "This point is part of the training data, so the models have seen it. "
            "Its score is not an independent test of the model.",
            "place": places.lookup(float(row["X"]), float(row["Y"])) if places else None,
        }

    def landslide_points(self) -> np.ndarray | None:
        """Projected coordinates of the recorded landslides, once loaded."""
        state = self._state
        if state is None:
            return None
        df = state["training"]
        return df.loc[df["label"] == 1, ["X", "Y"]].to_numpy(dtype="float64")
