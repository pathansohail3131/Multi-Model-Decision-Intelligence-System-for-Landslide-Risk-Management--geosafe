"""Model persistence with preprocessing welded on.

A model saved without its preprocessing is a loaded gun. It can be loaded, it
will happily produce predictions, and those predictions look entirely plausible
while being wrong. For raster output the failure is close to invisible — the
map still looks like a map, with plausible spatial structure, because the
terrain structure comes through regardless of whether the features were encoded
correctly.

The previous version of this project had exactly this shape of hazard: its SVM
and ANN needed separately applied preprocessors, and its meta-learner required
base predictions in one specific undocumented order (RF, XGB, CB, SVM, ANN).
Nothing enforced either.

Here a :class:`ModelBundle` is the only unit of persistence. It carries the
fitted encoder, the fitted estimator, the exact feature list, the categorical
column names, the operating threshold and the calibrator together. There is no
API for saving a bare estimator.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.models.estimators import prepare_features
from src.utils.logging_setup import get_logger

__all__ = ["ModelBundle", "save_bundle", "load_bundle"]

logger = get_logger(__name__)

#: Bumped when the bundle layout changes incompatibly, so an old artefact
#: fails loudly on load instead of being silently misread.
BUNDLE_FORMAT_VERSION = 1


@dataclass
class ModelBundle:
    """A fitted model together with everything needed to apply it correctly.

    Attributes
    ----------
    name : str
        Model name, e.g. ``"topographic"`` or ``"full"``.
    kind : str
        Estimator kind, e.g. ``"catboost"``.
    estimator : Any
        The fitted classifier.
    encoder : Any
        The fitted :class:`~src.features.encoders.FeatureEncoder`.
    features : list of str
        Exact feature columns, in order, that the estimator was trained on.
    categorical_features : list of str
        Which of those are categorical.
    threshold : float
        Operating threshold. ``None`` until Stage 5 derives it; never silently
        0.5.
    calibrator : Any, optional
        Fitted probability calibrator applied after ``predict_proba``. Required
        before any disagreement measure is computed across models, because
        variance over uncalibrated scores measures scaling differences rather
        than genuine disagreement.
    metadata : dict
        Provenance: seed, imbalance strategy, CV scheme, training counts,
        library versions, creation time.
    """

    name: str
    kind: str
    estimator: Any
    encoder: Any
    features: list[str]
    categorical_features: list[str] = field(default_factory=list)
    threshold: float | None = None
    calibrator: Any = None
    metadata: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    def predict_proba(self, raw: pd.DataFrame, calibrated: bool = True) -> np.ndarray:
        """Predict positive-class probability from *raw* conditioning factors.

        The input is raw factors, not encoded features. That is deliberate: it
        makes it impossible to feed the estimator anything the encoder has not
        processed.

        Parameters
        ----------
        raw : pandas.DataFrame
            Raw conditioning factors, as loaded from the CSV or sampled from
            the rasters.
        calibrated : bool, default True
            Apply the fitted calibrator if one is attached.

        Returns
        -------
        numpy.ndarray
            Probability of the positive class, one value per row.
        """
        encoded = self.encoder.transform(raw)
        missing = [c for c in self.features if c not in encoded.columns]
        if missing:
            raise ValueError(
                f"model {self.name!r} needs features absent after encoding: {missing}"
            )
        subset = encoded[self.features]
        prepared = prepare_features(subset, self.kind, self.categorical_features)
        probabilities = self.estimator.predict_proba(prepared)[:, 1]

        if calibrated and self.calibrator is not None:
            probabilities = self.calibrator.transform(probabilities)
        return probabilities

    def predict(self, raw: pd.DataFrame) -> np.ndarray:
        """Predict binary labels at the bundle's operating threshold.

        Raises
        ------
        ValueError
            If no threshold has been set. Defaulting to 0.5 here would quietly
            undo the threshold work in Stage 5, so it is refused.
        """
        if self.threshold is None:
            raise ValueError(
                f"model {self.name!r} has no operating threshold; "
                "Stage 5 must derive one before binary prediction is meaningful"
            )
        return (self.predict_proba(raw) >= self.threshold).astype(int)

    def summary(self) -> dict:
        """Return a JSON-serialisable description of the bundle."""
        return {
            "name": self.name,
            "kind": self.kind,
            "n_features": len(self.features),
            "features": list(self.features),
            "categorical_features": list(self.categorical_features),
            "threshold": self.threshold,
            "calibrated": self.calibrator is not None,
            "metadata": self.metadata,
        }


def save_bundle(bundle: ModelBundle, directory: Path) -> Path:
    """Serialise a bundle to disk.

    Writes two files: a joblib artefact holding the fitted objects, and a
    sidecar JSON summary that is readable without unpickling anything. The
    sidecar exists so that a model's provenance can be inspected in a text
    editor, in a diff, or on a machine that cannot load the pickle at all —
    which is precisely the situation the previous version's unloadable models
    left it in.

    Parameters
    ----------
    bundle : ModelBundle
        The fitted bundle.
    directory : Path
        Destination directory, created if absent.

    Returns
    -------
    Path
        Path to the joblib artefact.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    bundle.metadata.setdefault(
        "created_utc", datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    bundle.metadata.setdefault("format_version", BUNDLE_FORMAT_VERSION)
    bundle.metadata.setdefault("library_versions", _library_versions())

    path = directory / f"{bundle.name}.joblib"
    joblib.dump(bundle, path, compress=3)

    sidecar = directory / f"{bundle.name}.json"
    with sidecar.open("w", encoding="utf-8") as handle:
        json.dump(bundle.summary(), handle, indent=2, default=str)

    logger.info("Saved model bundle: %s (%.1f KB)", path, path.stat().st_size / 1024)
    return path


def load_bundle(path: Path) -> ModelBundle:
    """Load a bundle from disk.

    Parameters
    ----------
    path : Path
        Path to the joblib artefact.

    Returns
    -------
    ModelBundle
        The deserialised bundle.

    Raises
    ------
    ValueError
        If the artefact's format version is newer than this code understands.
    """
    bundle = joblib.load(Path(path))
    version = bundle.metadata.get("format_version", 0)
    if version > BUNDLE_FORMAT_VERSION:
        raise ValueError(
            f"bundle {path} was written in format v{version}; this code reads "
            f"up to v{BUNDLE_FORMAT_VERSION}. Check out the matching commit."
        )
    logger.info("Loaded model bundle: %s (%s)", bundle.name, path)
    return bundle


def _library_versions() -> dict[str, str]:
    """Capture versions of libraries whose pickles are version-sensitive."""
    versions: dict[str, str] = {}
    for module_name in ("sklearn", "catboost", "xgboost", "numpy", "pandas"):
        try:
            module = __import__(module_name)
            versions[module_name] = getattr(module, "__version__", "unknown")
        except ImportError:
            continue
    return versions
