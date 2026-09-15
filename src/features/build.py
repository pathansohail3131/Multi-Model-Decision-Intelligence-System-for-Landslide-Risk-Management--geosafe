"""Stage 2: build the encoded feature matrix and the thematic groups.

This stage is the single place the dataset is turned into model features, and
:func:`prepare_dataset` is the single function every later stage calls to get
that dataset. Having one entry point matters more than it might seem: if
Stage 4 encoded features slightly differently from Stage 7, the models would be
trained and applied on different representations, and nothing would raise an
error — the district map would simply be wrong.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd

from src.data.loaders import load_training_data
from src.features.encoders import build_encoder
from src.features.themes import BENCHMARK_NAME, all_theme_features, validate_partition
from src.utils.logging_setup import get_logger

__all__ = ["Dataset", "prepare_dataset", "run_feature_stage", "ENCODER_FILENAME"]

logger = get_logger(__name__)

#: Filename of the fitted encoder inside the models directory.
ENCODER_FILENAME = "feature_encoder.joblib"


@dataclass
class Dataset:
    """The modelling dataset in the form every later stage consumes.

    Attributes
    ----------
    raw : pandas.DataFrame
        Cleaned raw conditioning factors, including coordinates and the
        blocked metadata columns (retained for reporting, never modelled).
    X : pandas.DataFrame
        Encoded features.
    y : numpy.ndarray
        Binary target.
    coords : numpy.ndarray, shape (n, 2)
        Projected coordinates in metres, for spatial blocking.
    encoder : FeatureEncoder
        The fitted encoder.
    theme_features : dict
        Model name mapped to its feature list, including ``"full"``.
    categorical_features : list of str
        Categorical column names among the encoded features.
    """

    raw: pd.DataFrame
    X: pd.DataFrame
    y: np.ndarray
    coords: np.ndarray
    encoder: object
    theme_features: dict
    categorical_features: list[str]

    def theme_X(self, theme: str) -> pd.DataFrame:
        """Return the encoded features for one model.

        Parameters
        ----------
        theme : str
            Theme name or ``"full"``.

        Returns
        -------
        pandas.DataFrame
            That model's feature columns.
        """
        return self.X[self.theme_features[theme]]

    def theme_categoricals(self, theme: str) -> list[str]:
        """Return the categorical columns belonging to one model."""
        return [c for c in self.categorical_features if c in self.theme_features[theme]]


def prepare_dataset(cfg) -> Dataset:
    """Load, clean and encode the modelling dataset.

    Parameters
    ----------
    cfg : Config
        Resolved project configuration.

    Returns
    -------
    Dataset
        Everything downstream stages need, consistently encoded.
    """
    raw, _ = load_training_data(cfg, drop_invalid=True)

    target = cfg.get("features", "target", default="label")
    x_col, y_col = cfg.get("features", "coord_columns", default=["X", "Y"])

    encoder = build_encoder(cfg)
    X = encoder.fit_transform(raw)
    y = raw[target].to_numpy().astype(int)
    coords = raw[[x_col, y_col]].to_numpy(dtype="float64")

    themes = all_theme_features(cfg, list(X.columns))
    return Dataset(
        raw=raw,
        X=X,
        y=y,
        coords=coords,
        encoder=encoder,
        theme_features=themes,
        categorical_features=list(encoder.categorical_features_),
    )


def run_feature_stage(cfg) -> dict:
    """Stage 2 entry point: build features, validate the partition, persist.

    Parameters
    ----------
    cfg : Config
        Resolved project configuration.

    Returns
    -------
    dict
        Report written to ``outputs/reports/features.json``.
    """
    logger.info("Building feature matrix")

    partition = validate_partition(cfg)
    if not partition["is_partition"]:
        # Not fatal — the pipeline can still run — but it invalidates the
        # central claim, so it must be impossible to miss in the log.
        logger.error(
            "Thematic feature subsets do not partition the conditioning "
            "factors. The error-decorrelation argument does not hold as "
            "configured; see docs/methodology.md section 3.1."
        )

    dataset = prepare_dataset(cfg)

    # Persist the encoder separately as well as inside each model bundle, so
    # that inference can encode a raster block once and reuse it across all
    # five models rather than re-encoding per model.
    encoder_path = cfg.models_dir / ENCODER_FILENAME
    encoder_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(dataset.encoder, encoder_path, compress=3)
    logger.info("Saved fitted encoder: %s", encoder_path)

    # Cache the encoded matrix for inspection and diffing. CSV rather than a
    # binary format deliberately: this file exists to be opened and read by a
    # human checking what the model actually saw, and it is not on any
    # pipeline critical path (later stages re-derive it via prepare_dataset).
    encoded_path = cfg.reports_dir / "encoded_features.csv"
    cached = dataset.X.copy()
    cached["label"] = dataset.y
    cached["X_coord"] = dataset.coords[:, 0]
    cached["Y_coord"] = dataset.coords[:, 1]
    cached.to_csv(encoded_path, index=False)
    logger.info("Cached encoded features: %s", encoded_path)

    report = {
        "n_samples": int(len(dataset.X)),
        "n_positive": int(dataset.y.sum()),
        "n_features": int(dataset.X.shape[1]),
        "feature_names": list(dataset.X.columns),
        "categorical_features": dataset.categorical_features,
        "categorical_levels": {
            column: list(map(str, levels))
            for column, levels in dataset.encoder.categories_.items()
        },
        "theme_partition": partition,
        "model_feature_counts": {
            name: len(features) for name, features in dataset.theme_features.items()
        },
        "model_features": {
            name: list(features) for name, features in dataset.theme_features.items()
        },
        "encoder_path": str(encoder_path),
        "encoded_cache": str(encoded_path) if encoded_path else None,
    }

    logger.info(
        "Encoded %d samples into %d features (%d categorical)",
        report["n_samples"],
        report["n_features"],
        len(report["categorical_features"]),
    )
    for name, count in report["model_feature_counts"].items():
        marker = " (benchmark)" if name == BENCHMARK_NAME else ""
        logger.info("  %-16s %2d features%s", name, count, marker)

    out_path = cfg.reports_dir / "features.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)
    logger.info("Feature report written: %s", out_path)

    return report
