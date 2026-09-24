"""Model bundles: loading, feature contract, probability ranges and the ensemble rule."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.themes import BENCHMARK_NAME
from src.models.combiners import (
    confidence_from_disagreement,
    disagreement,
    full_plus_thematic_mean,
    mean_combine,
)
from tests.conftest import requires_bundles

pytestmark = requires_bundles


def test_every_bundle_carries_its_preprocessing(bundles):
    """A model without its encoder would silently score wrong features."""
    for name, bundle in bundles.items():
        assert bundle.encoder is not None, f"{name} has no encoder"
        assert bundle.features, f"{name} has no feature list"
        assert bundle.calibrator is not None, f"{name} has no calibrator (run Stage 5)"
        assert bundle.threshold is not None, f"{name} has no operating threshold (run Stage 5)"


def test_all_bundles_share_one_threshold(bundles):
    """Different thresholds would mean the bundles came from different runs."""
    thresholds = {round(float(b.threshold), 10) for b in bundles.values()}
    assert len(thresholds) == 1


def test_thematic_models_see_only_their_own_theme(cfg, bundles):
    """The architecture's central claim: the themes partition the factors."""
    seen: set[str] = set()
    for theme in cfg.theme_names:
        factors = set(cfg.get("themes", theme, "factors"))
        assert not (factors & seen), f"{theme} overlaps an earlier theme"
        seen |= factors
        encoded = set(bundles[theme].features)
        assert encoded == set(cfg.theme_features(theme)) & encoded
        assert encoded, f"{theme} model has no features"
    assert seen == set(cfg.raster_factors)
    assert len(bundles[BENCHMARK_NAME].features) >= len(cfg.raster_factors)


def test_predictions_are_probabilities(cfg, bundles, training_data):
    frame = training_data[cfg.raster_factors].head(24).astype("float64")
    for name, bundle in bundles.items():
        probabilities = bundle.predict_proba(frame)
        assert probabilities.shape == (24,), f"{name} returned the wrong shape"
        assert np.all(np.isfinite(probabilities)), f"{name} returned non-finite values"
        assert np.all((probabilities >= 0) & (probabilities <= 1)), f"{name} left [0, 1]"


def test_feature_order_is_enforced_not_assumed(cfg, bundles, training_data):
    """Shuffled input columns must not change the score: the encoder reorders them."""
    frame = training_data[cfg.raster_factors].head(8).astype("float64")
    shuffled = frame[list(reversed(cfg.raster_factors))]
    bundle = bundles[BENCHMARK_NAME]
    assert bundle.predict_proba(frame) == pytest.approx(bundle.predict_proba(shuffled))


def test_encoder_output_is_deterministic(bundles, cfg, training_data):
    frame = training_data[cfg.raster_factors].head(16).astype("float64")
    encoder = bundles[BENCHMARK_NAME].encoder
    first = encoder.transform(frame)
    second = encoder.transform(frame)
    assert list(first.columns) == list(second.columns)
    pd.testing.assert_frame_equal(first, second)


def test_prediction_is_repeatable(bundles, cfg, training_data):
    frame = training_data[cfg.raster_factors].head(8).astype("float64")
    bundle = bundles[BENCHMARK_NAME]
    assert bundle.predict_proba(frame) == pytest.approx(bundle.predict_proba(frame))


def test_binary_prediction_uses_the_operating_threshold(bundles, cfg, training_data):
    frame = training_data[cfg.raster_factors].head(32).astype("float64")
    bundle = bundles[BENCHMARK_NAME]
    expected = (bundle.predict_proba(frame) >= bundle.threshold).astype(int)
    assert np.array_equal(bundle.predict(frame), expected)


def test_production_blend_is_the_documented_rule():
    full = np.array([0.8, 0.2])
    thematic = np.array([[0.6, 0.4, 0.2, 0.0], [0.4, 0.4, 0.4, 0.4]])
    blended = full_plus_thematic_mean(full, thematic)
    assert blended == pytest.approx([0.5 * (0.8 + 0.3), 0.5 * (0.2 + 0.4)])
    assert mean_combine(thematic) == pytest.approx([0.3, 0.4])


def test_disagreement_and_confidence_are_consistent():
    identical = np.array([[0.4, 0.4, 0.4, 0.4]])
    spread = np.array([[0.0, 0.3, 0.6, 0.9]])
    assert disagreement(identical, "std")[0] == pytest.approx(0.0)
    assert disagreement(spread, "std")[0] > 0
    both = disagreement(np.vstack([identical, spread]), "std")
    confidence = confidence_from_disagreement(both)
    assert confidence[0] == pytest.approx(1.0)  # full agreement -> full confidence
    assert 0.0 <= confidence[1] < confidence[0]


def test_ensemble_output_stays_in_range(cfg, bundles, training_data):
    frame = training_data[cfg.raster_factors].head(50).astype("float64")
    thematic = np.column_stack([bundles[t].predict_proba(frame) for t in cfg.theme_names])
    blended = full_plus_thematic_mean(bundles[BENCHMARK_NAME].predict_proba(frame), thematic)
    assert blended.shape == (50,)
    assert np.all((blended >= 0) & (blended <= 1))


def test_recorded_landslides_score_above_non_landslides_on_average(cfg, bundles, training_data):
    """A sanity check on the trained artefacts, not a performance claim."""
    sample = training_data.groupby("label", group_keys=False).head(60)
    frame = sample[cfg.raster_factors].astype("float64")
    scores = bundles[BENCHMARK_NAME].predict_proba(frame)
    positives = scores[sample["label"].to_numpy() == 1]
    negatives = scores[sample["label"].to_numpy() == 0]
    assert positives.mean() > negatives.mean()
