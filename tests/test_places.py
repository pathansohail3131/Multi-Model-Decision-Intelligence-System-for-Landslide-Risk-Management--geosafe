"""Place-name lookup: known locations, the fallbacks, speed and the API fields."""

from __future__ import annotations

import time

import pytest
from pyproj import Transformer

from api.app.services import places as places_module
from api.app.services.places import DATA_DIR, PlaceIndex, get_place_index
from tests.conftest import requires_bundles

pytestmark = pytest.mark.skipif(
    not (DATA_DIR / "villages.geojson").exists(), reason="place-name data not available"
)

TO_GRID = Transformer.from_crs("EPSG:4326", "EPSG:32643", always_xy=True)


@pytest.fixture(scope="module")
def index():
    return PlaceIndex()


def lookup(index, lon, lat):
    return index.lookup(*TO_GRID.transform(lon, lat))


def test_sirsi(index):
    place = lookup(index, 74.835, 14.62)
    assert place["taluk"] == "Sirsi"
    assert place["district"] == "Uttara Kannada"
    assert place["state"] == "Karnataka"
    assert place["sublabel"] == "Sirsi taluk, Uttara Kannada"


def test_ankola(index):
    assert lookup(index, 74.38, 14.66)["taluk"] == "Ankola"


def test_sea_is_outside_every_village(index):
    place = lookup(index, 73.9, 14.6)
    assert place["label"] == "Outside mapped villages"
    assert place["village"] is None
    assert place["taluk"] is None
    assert place["inside_village"] is False


def test_taluk_outlines_cover_all_eleven(index):
    outlines = index.taluk_outlines()
    assert len(outlines) == 11
    assert "Joida (Supa)" in {o["name"] for o in outlines}
    for outline in outlines:
        assert outline["rings"] and len(outline["label"]) == 2
        assert index.lookup(*outline["label"])["taluk"] is not None


def test_nearest_town_direction_is_measured_from_the_town(index):
    town_x, town_y = index._town_xy[index._town_names.index("Sirsi")]
    place = index.lookup(float(town_x) + 2000, float(town_y) - 2000)
    assert place["nearest_town"]["name"] == "Sirsi"
    assert place["nearest_town"]["direction_from_town"] == "SE"
    assert place["nearest_town"]["distance_m"] == pytest.approx(2828.4, abs=0.1)


def test_census_suffix_becomes_a_town_flag(index):
    names = [v["village"] for v in index._villages]
    assert not any(name.endswith(("(CMC)", "(TMC)", "(TP)", "(CT)", "+ OG)")) for name in names)
    assert any(v["town"] for v in index._villages)


def test_town_label(index):
    town = next(i for i, v in enumerate(index._villages) if v["town"])
    point = index._village_tree.geometries[town].representative_point()
    place = index.lookup(point.x, point.y)
    assert place["is_town"] is True
    assert place["label"] == f"{place['village']} town"


def test_not_finite_coordinates_do_not_raise(index):
    assert index.lookup(float("nan"), 1.0)["label"] == "Outside mapped villages"


def test_lookup_is_under_a_millisecond_on_average(index):
    left, bottom = TO_GRID.transform(74.1, 13.9)
    right, top = TO_GRID.transform(75.0, 15.5)
    n = 2000
    points = [
        (left + (right - left) * (i % 50) / 50, bottom + (top - bottom) * (i % 40) / 40) for i in range(n)
    ]
    started = time.perf_counter()
    for x, y in points:
        index.lookup(x, y)
    assert (time.perf_counter() - started) / n < 0.001


def test_missing_data_returns_none_and_does_not_raise(monkeypatch, tmp_path):
    monkeypatch.setattr(places_module, "_index", None)
    monkeypatch.setattr(places_module, "_loaded", False)
    monkeypatch.setattr(places_module.PlaceIndex.__init__, "__defaults__", (tmp_path,))
    assert get_place_index() is None


@requires_bundles
def test_api_responses_carry_place_fields(client):
    sample = client.get("/api/samples/random", params={"label": 1}).json()
    assert sample["place"]["district"] == "Uttara Kannada"

    site = client.get("/api/district/sample", params={"x": sample["x"], "y": sample["y"]}).json()
    assert site["place"]["label"] == sample["place"]["label"]

    assert len(client.get("/api/district/meta").json()["taluks"]) == 11
