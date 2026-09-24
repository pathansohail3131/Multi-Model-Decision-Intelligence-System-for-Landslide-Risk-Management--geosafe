#!/usr/bin/env python3
"""Score a CSV of sites with the trained bundles, without starting the API.

Each input row must carry the 14 raw conditioning factors (the column names in
``config.yaml`` under ``rasters``); extra columns are carried through to the
output untouched, so an id or coordinate column survives the round trip.

Usage
-----
Run from the project root. Score a file::

    python -m scripts.predict sites.csv --output scored.csv

Sample the district rasters at coordinates instead of supplying factors::

    python -m scripts.predict coords.csv --from-rasters --x-column X --y-column Y

Print a few rows of the training data as a template::

    python -m scripts.predict --template 3

The output adds: the per-model calibrated probabilities, the production
susceptibility, its class, the flag at the operating threshold, cross-theme
disagreement and, when Stage 5 reports are present, confidence and the planner
decision.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Executing this file directly puts scripts/ on sys.path rather than the project
# root, so `src` would not resolve. `python -m scripts.predict` is the documented
# form; this keeps `python scripts/predict.py` working as well.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import load_config  # noqa: E402
from src.data.loaders import load_training_data, sample_factors_at_points  # noqa: E402
from src.features.themes import BENCHMARK_NAME  # noqa: E402
from src.inference.raster import susceptibility_classes  # noqa: E402
from src.models.bundle import load_bundle  # noqa: E402
from src.models.combiners import (
    confidence_from_disagreement,
    disagreement,
    full_plus_thematic_mean,
)
from src.utils.logging_setup import configure_logging, get_logger  # noqa: E402

logger = get_logger("scripts.predict")


def _load_bundles(cfg):
    names = [*cfg.theme_names, BENCHMARK_NAME]
    bundles = {}
    for name in names:
        path = cfg.models_dir / f"{name}.joblib"
        if not path.exists():
            raise SystemExit(
                f"Model bundle not found: {path}\n"
                "Train the models first: python run_pipeline.py stage 1 --to 5"
            )
        bundles[name] = load_bundle(path)
    missing = [n for n, b in bundles.items() if b.threshold is None or b.calibrator is None]
    if missing:
        raise SystemExit(
            f"Bundles {missing} carry no calibrator or threshold. Run: python run_pipeline.py stage 5"
        )
    return bundles


def _confidence_reference(cfg) -> tuple[float | None, float | None]:
    """Stage 5 scaling values: the disagreement ceiling and the high/low split."""
    import json

    oof = cfg.reports_dir / "oof_calibrated.csv"
    report = cfg.reports_dir / "calibration_threshold.json"
    if not (oof.exists() and report.exists()):
        return None, None
    ceiling = float(np.nanmax(pd.read_csv(oof, usecols=["disagreement"]).to_numpy()))
    split = json.loads(report.read_text(encoding="utf-8")).get("decision_quadrants", {}).get(
        "confidence_split_at"
    )
    return ceiling, (float(split) if split is not None else None)


def score(cfg, frame: pd.DataFrame) -> pd.DataFrame:
    """Score raw conditioning factors. Returns the input with result columns added."""
    bundles = _load_bundles(cfg)
    themes = list(cfg.theme_names)
    threshold = float(bundles[BENCHMARK_NAME].threshold)
    ceiling, split = _confidence_reference(cfg)

    missing = [f for f in cfg.raster_factors if f not in frame.columns]
    if missing:
        raise SystemExit(f"Input is missing required factor column(s): {missing}")

    raw = frame[cfg.raster_factors].astype("float64")
    usable = raw.notna().all(axis=1)
    if not usable.any():
        raise SystemExit("No row has all 14 conditioning factors present.")
    if (~usable).any():
        logger.warning("%d row(s) have missing factors and are left unscored", int((~usable).sum()))

    subset = raw.loc[usable]
    calibrated = {name: bundle.predict_proba(subset) for name, bundle in bundles.items()}
    thematic = np.column_stack([calibrated[t] for t in themes])
    susceptibility = full_plus_thematic_mean(calibrated[BENCHMARK_NAME], thematic)
    spread = disagreement(thematic, "std")

    out = frame.copy()
    for name in themes + [BENCHMARK_NAME]:
        out.loc[usable, f"p_{name}"] = calibrated[name]
    out.loc[usable, "susceptibility"] = susceptibility
    out.loc[usable, "disagreement"] = spread
    out.loc[usable, "above_threshold"] = susceptibility >= threshold

    bands = susceptibility_classes(threshold)
    out.loc[usable, "susceptibility_class"] = [
        next(k for k, (low, high) in bands.items() if low <= value < high) for value in susceptibility
    ]

    if ceiling and split is not None:
        confidence = confidence_from_disagreement(spread, max_possible=ceiling)
        out.loc[usable, "confidence"] = confidence
        high_susc = susceptibility >= threshold
        high_conf = confidence >= split
        out.loc[usable, "decision"] = np.where(
            high_susc & high_conf,
            "act_now",
            np.where(high_susc, "survey_first", np.where(high_conf, "deprioritise", "monitor")),
        )
    else:
        logger.warning(
            "Stage 5 reports not found; confidence and the planner decision are omitted. "
            "Run: python run_pipeline.py stage 5"
        )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", nargs="?", type=Path, help="CSV of sites to score")
    parser.add_argument("-o", "--output", type=Path, help="Where to write the scored CSV")
    parser.add_argument(
        "--from-rasters",
        action="store_true",
        help="Read the 14 factors from the district rasters at each row's coordinates",
    )
    parser.add_argument("--x-column", default="X", help="Easting column when --from-rasters is used")
    parser.add_argument("--y-column", default="Y", help="Northing column when --from-rasters is used")
    parser.add_argument(
        "--template", type=int, metavar="N", help="Print N rows of real training data as a template and exit"
    )
    args = parser.parse_args(argv)

    cfg = load_config()
    configure_logging(level=str(cfg.get("logging", "level", default="INFO")), run_name="predict", to_file=False)

    if args.template:
        frame, _ = load_training_data(cfg)
        print(frame[cfg.raster_factors].head(args.template).to_csv(index=False))
        return 0

    if args.input is None:
        parser.error("an input CSV is required (or use --template)")
    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")

    frame = pd.read_csv(args.input)
    if args.from_rasters:
        for column in (args.x_column, args.y_column):
            if column not in frame.columns:
                raise SystemExit(f"--from-rasters needs a {column!r} column")
        sampled = sample_factors_at_points(
            cfg, frame, cfg.raster_factors, x_col=args.x_column, y_col=args.y_column
        )
        frame = pd.concat([frame.drop(columns=sampled.columns, errors="ignore"), sampled], axis=1)

    scored = score(cfg, frame)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        scored.to_csv(args.output, index=False)
        logger.info("Wrote %d scored row(s): %s", len(scored), args.output)
    else:
        columns = [c for c in scored.columns if c.startswith("p_")] + [
            "susceptibility",
            "susceptibility_class",
            "above_threshold",
        ]
        columns += [c for c in ("confidence", "decision") if c in scored.columns]
        print(scored[columns].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
