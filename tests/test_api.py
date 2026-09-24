"""Prediction API behaviour: endpoints, response shapes and agreement with the bundles."""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import requires_bundles

pytestmark = requires_bundles


def test_health_reports_ready(client, cfg):
    body = client.get("/api/health").json()
    assert body["status"] == "ready"
    assert set(body["models"]) == {*cfg.theme_names, "full"}
    assert body["district_map"] in {"idle", "loading", "ready", "error"}


def test_openapi_schema_lists_every_route(client):
    paths = client.get("/openapi.json").json()["paths"]
    for route in [
        "/api/health",
        "/api/metadata",
        "/api/predict",
        "/api/sensitivity",
        "/api/samples/random",
        "/api/district/meta",
        "/api/district/sample",
    ]:
        assert route in paths, f"{route} missing from the OpenAPI schema"


def test_metadata_describes_all_factors(client, cfg):
    body = client.get("/api/metadata").json()
    assert {f["name"] for f in body["factors"]} == set(cfg.raster_factors)
    assert body["threshold"]["value"] > 0
    assert len(body["classes"]) == 5
    assert body["dataset"]["n_samples"] > 0


def test_predict_matches_the_bundles_directly(client, cfg, bundles, landslide_row):
    """The API must return exactly what the saved bundles compute."""
    import pandas as pd

    from src.models.combiners import full_plus_thematic_mean

    response = client.post("/api/predict", json={"factors": landslide_row, "explain": True})
    assert response.status_code == 200
    body = response.json()

    frame = pd.DataFrame([landslide_row], columns=cfg.raster_factors).astype("float64")
    thematic = np.column_stack([bundles[t].predict_proba(frame) for t in cfg.theme_names])
    expected = float(full_plus_thematic_mean(bundles["full"].predict_proba(frame), thematic)[0])

    assert body["prediction"]["susceptibility"] == pytest.approx(expected, abs=1e-6)
    assert body["prediction"]["above_threshold"] == (expected >= bundles["full"].threshold)


def test_prediction_response_shape(client, landslide_row):
    body = client.post("/api/predict", json={"factors": landslide_row}).json()
    prediction = body["prediction"]
    for key in ["susceptibility", "threshold", "above_threshold", "class", "full_model", "disagreement"]:
        assert key in prediction
    assert 0.0 <= prediction["susceptibility"] <= 1.0
    assert {m["name"] for m in body["models"]} == {"topographic", "hydrological", "geological", "anthropogenic", "full"}
    assert all(0.0 <= m["probability"] <= 1.0 for m in body["models"])
    assert body["timing_ms"]["inference"] >= 0


def test_shap_values_sum_to_the_model_output(client, landslide_row):
    body = client.post("/api/predict", json={"factors": landslide_row}).json()
    full = next(m for m in body["models"] if m["name"] == "full")
    explanation = body["explanations"]["full"]
    total = explanation["base_log_odds"] + sum(f["shap"] for f in explanation["features"])
    assert 1 / (1 + np.exp(-total)) == pytest.approx(full["raw_probability"], abs=1e-4)


def test_explanations_can_be_skipped(client, landslide_row):
    body = client.post("/api/predict", json={"factors": landslide_row, "explain": False}).json()
    assert body["explanations"] is None
    assert body["timing_ms"]["explanation"] is None


def test_sensitivity_includes_the_current_value(client, landslide_row):
    response = client.post(
        "/api/sensitivity", json={"factors": landslide_row, "factor": "dist_road", "points": 11}
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["values"]) == len(body["susceptibility"])
    assert body["current_value"] in body["values"]
    assert all(0.0 <= v <= 1.0 for v in body["susceptibility"])
    assert set(body["thematic"]) == {"topographic", "hydrological", "geological", "anthropogenic"}


def test_sensitivity_rejects_an_unknown_factor(client, landslide_row):
    response = client.post(
        "/api/sensitivity", json={"factors": landslide_row, "factor": "not_a_factor"}
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_input"


def test_random_sample_returns_a_real_labelled_row(client, cfg):
    body = client.get("/api/samples/random", params={"label": 0}).json()
    assert body["label"] == 0
    assert set(body["factors"]) == set(cfg.raster_factors)
    assert "training data" in body["note"]


def test_district_meta_and_sample(client, cfg):
    meta = client.get("/api/district/meta").json()
    assert meta["status"] in {"idle", "loading", "ready", "error"}
    if meta["status"] != "ready":
        pytest.skip("district map still building")

    left, bottom, right, top = cfg.get("grid", "bounds")
    body = client.get(
        "/api/district/sample", params={"x": (left + right) / 2, "y": (bottom + top) / 2}
    ).json()
    assert "inside" in body
    if body["inside"]:
        assert set(body["factors"]) == set(cfg.raster_factors)


def test_district_sample_outside_the_grid_is_reported(client):
    body = client.get("/api/district/sample", params={"x": 0, "y": 0}).json()
    assert body["inside"] is False
    assert body["issues"]


def test_unknown_district_layer_returns_404(client):
    response = client.get("/api/district/image/not_a_layer.png")
    assert response.status_code in {404, 503}
    assert response.json()["error"]["code"] in {"layer_not_found", "map_loading"}
