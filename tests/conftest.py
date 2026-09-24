"""Shared fixtures.

Tests that need trained artefacts are skipped, not failed, when the pipeline
has not been run on this machine: the models are large and live outside the
repository, so a clean checkout legitimately has none.
"""

from __future__ import annotations

import time

import pytest

from src.config import load_config
from src.features.themes import BENCHMARK_NAME

CONFIG = load_config()
MODEL_NAMES = [*CONFIG.theme_names, BENCHMARK_NAME]
BUNDLE_PATHS = [CONFIG.models_dir / f"{name}.joblib" for name in MODEL_NAMES]

HAVE_BUNDLES = all(path.exists() for path in BUNDLE_PATHS)
HAVE_TRAINING_CSV = CONFIG.training_csv.exists()
HAVE_RASTERS = not CONFIG.missing_rasters() if CONFIG.processed_dir.exists() else False

requires_bundles = pytest.mark.skipif(
    not (HAVE_BUNDLES and HAVE_TRAINING_CSV),
    reason="trained model bundles or training data not available (run the pipeline through Stage 5)",
)
requires_rasters = pytest.mark.skipif(
    not HAVE_RASTERS, reason="conditioning-factor rasters not available"
)


@pytest.fixture(scope="session")
def cfg():
    """Resolved project configuration."""
    return CONFIG


@pytest.fixture(scope="session")
def bundles(cfg):
    """Every trained bundle, loaded once."""
    from src.models.bundle import load_bundle

    return {name: load_bundle(cfg.models_dir / f"{name}.joblib") for name in MODEL_NAMES}


@pytest.fixture(scope="session")
def training_data(cfg):
    """The cleaned training dataset."""
    from src.data.loaders import load_training_data

    frame, _ = load_training_data(cfg)
    return frame


@pytest.fixture(scope="session")
def landslide_row(cfg, training_data):
    """Raw conditioning factors of one recorded landslide, as an API payload."""
    row = training_data[training_data["label"] == 1].iloc[0]
    return {factor: float(row[factor]) for factor in cfg.raster_factors}


@pytest.fixture(scope="session")
def client():
    """TestClient with the model bundles fully loaded."""
    from fastapi.testclient import TestClient

    from api.app.dependencies import prediction_service
    from api.app.main import app

    with TestClient(app) as test_client:
        deadline = time.time() + 180
        while prediction_service.status == "loading" and time.time() < deadline:
            time.sleep(0.5)
        if prediction_service.status != "ready":
            pytest.skip(f"prediction service not ready: {prediction_service.error}")
        yield test_client


@pytest.fixture(scope="session")
def api_sample(client):
    """A real training row fetched through the API."""
    response = client.get("/api/samples/random", params={"label": 1})
    assert response.status_code == 200
    return response.json()
