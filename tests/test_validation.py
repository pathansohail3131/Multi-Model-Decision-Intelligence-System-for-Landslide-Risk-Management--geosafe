"""Input validation.

The rule the API implements: physically impossible values are rejected, values
that are merely outside the *training* range are accepted and flagged, because
refusing them would hide that the models are extrapolating.
"""

from __future__ import annotations

import pytest

from tests.conftest import requires_bundles

pytestmark = requires_bundles


def test_missing_fields_are_listed_individually(client):
    response = client.post("/api/predict", json={"factors": {"slope": 20.0}})
    assert response.status_code == 422
    body = response.json()
    assert body["error"]["code"] == "invalid_input"
    missing = {item["field"] for item in body["error"]["fields"]}
    assert "elevation" in missing and "rainfall" in missing


def test_wrong_type_is_rejected(client, landslide_row):
    factors = {**landslide_row, "slope": "steep"}
    response = client.post("/api/predict", json={"factors": factors})
    assert response.status_code == 422
    assert any(item["field"] == "slope" for item in response.json()["error"]["fields"])


@pytest.mark.parametrize(
    ("factor", "value"),
    [("slope", 120.0), ("slope", -5.0), ("aspect", 400.0), ("NDVI", 2.0), ("elevation", -10.0)],
)
def test_physically_impossible_values_are_rejected(client, landslide_row, factor, value):
    response = client.post("/api/predict", json={"factors": {**landslide_row, factor: value}})
    assert response.status_code == 422
    assert any(item["field"] == factor for item in response.json()["error"]["fields"])


def test_nan_is_rejected(client, landslide_row):
    # JSON NaN is non-standard; the API must not accept it as a number.
    payload = '{"factors": %s}' % str({**landslide_row, "slope": float("nan")}).replace("'", '"')
    response = client.post(
        "/api/predict", content=payload.encode(), headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422


def test_unknown_categorical_code_is_rejected(client, landslide_row):
    response = client.post("/api/predict", json={"factors": {**landslide_row, "LULC": 7}})
    assert response.status_code == 422
    assert any(item["field"] == "LULC" for item in response.json()["error"]["fields"])


def test_unexpected_field_is_rejected(client, landslide_row):
    response = client.post("/api/predict", json={"factors": {**landslide_row, "elevation_m": 100}})
    assert response.status_code == 422


def test_boundary_values_are_accepted(client, landslide_row):
    """The physical limits themselves are valid inputs."""
    factors = {**landslide_row, "slope": 90.0, "aspect": 360.0, "NDVI": -1.0, "elevation": 0.0}
    response = client.post("/api/predict", json={"factors": factors, "explain": False})
    assert response.status_code == 200
    assert 0.0 <= response.json()["prediction"]["susceptibility"] <= 1.0


def test_out_of_training_range_is_flagged_not_rejected(client, landslide_row):
    response = client.post(
        "/api/predict", json={"factors": {**landslide_row, "elevation": 2500.0}, "explain": False}
    )
    assert response.status_code == 200
    kinds = {(w["factor"], w["kind"]) for w in response.json()["warnings"]}
    assert ("elevation", "outside_training_range") in kinds


def test_flat_cell_note_is_returned(client, landslide_row):
    body = client.post(
        "/api/predict", json={"factors": {**landslide_row, "slope": 0.0}, "explain": False}
    ).json()
    assert any(w["kind"] == "flat_cell" for w in body["warnings"])


def test_sparse_categorical_level_is_flagged(client, landslide_row):
    """LULC 60 (bare/sparse) has 12 training samples; the response must say so."""
    body = client.post(
        "/api/predict", json={"factors": {**landslide_row, "LULC": 60}, "explain": False}
    ).json()
    assert any(w["kind"] == "sparse_level" for w in body["warnings"])


def test_sensitivity_point_count_is_bounded(client, landslide_row):
    response = client.post(
        "/api/sensitivity", json={"factors": landslide_row, "factor": "slope", "points": 500}
    )
    assert response.status_code == 422
