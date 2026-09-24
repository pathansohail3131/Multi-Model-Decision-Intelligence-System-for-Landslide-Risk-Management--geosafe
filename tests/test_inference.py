"""Inference path: susceptibility classes, raster sampling and the produced district map.

The district run itself (10.9 M pixels) is a pipeline stage, not a test. What is
tested here is the contract around it: the class bands, the raster→features→score
path on a handful of real cells, and the shape of the written GeoTIFF when one
exists.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.inference.raster import susceptibility_classes, usable_themes
from tests.conftest import requires_bundles, requires_rasters


def test_class_bands_tile_the_unit_interval():
    bands = susceptibility_classes(0.1374)
    assert list(bands) == ["very_low", "low", "moderate", "high", "very_high"]
    lows = [low for low, _ in bands.values()]
    highs = [high for _, high in bands.values()]
    assert lows[0] == 0.0
    assert highs[-1] > 1.0  # upper band is inclusive of 1.0
    for previous_high, next_low in zip(highs, lows[1:]):
        assert previous_high == next_low, "class bands must not overlap or leave gaps"


def test_threshold_is_a_class_boundary():
    """"High" must mean "flagged at the operating point"."""
    threshold = 0.1374
    bands = susceptibility_classes(threshold)
    assert bands["high"][0] == threshold
    assert bands["moderate"][1] == threshold


def assign(value: float, threshold: float = 0.1374) -> str:
    return next(k for k, (low, high) in susceptibility_classes(threshold).items() if low <= value < high)


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0.0, "very_low"), (0.049, "very_low"), (0.05, "low"), (0.1374, "high"), (0.4, "very_high"), (1.0, "very_high")],
)
def test_class_assignment_at_boundaries(value, expected):
    assert assign(value) == expected


@requires_rasters
def test_all_themes_are_usable_when_rasters_are_present(cfg):
    themes, full_available = usable_themes(cfg)
    assert themes == cfg.theme_names
    assert full_available is True


@requires_bundles
@requires_rasters
def test_raster_sampling_feeds_the_models(cfg, bundles, training_data):
    """The geospatial path: sample rasters at coordinates, then score those cells."""
    from src.data.loaders import sample_factors_at_points
    from src.models.combiners import full_plus_thematic_mean

    points = training_data.head(5)
    sampled = sample_factors_at_points(cfg, points, cfg.raster_factors)
    assert list(sampled.columns) == list(cfg.raster_factors)
    assert len(sampled) == 5

    usable = sampled.dropna()
    if usable.empty:
        pytest.skip("sampled points fall on nodata")

    thematic = np.column_stack([bundles[t].predict_proba(usable) for t in cfg.theme_names])
    scores = full_plus_thematic_mean(bundles["full"].predict_proba(usable), thematic)
    assert np.all((scores >= 0) & (scores <= 1))
    assert np.all(np.isfinite(scores))


@requires_rasters
def test_district_output_matches_the_configured_grid(cfg):
    """If Stage 7 has run, its raster must sit on the same grid as the inputs."""
    import rasterio
    from rasterio.windows import Window

    path = cfg.rasters_dir / "susceptibility.tif"
    if not path.exists():
        pytest.skip("district susceptibility raster not produced yet (Stage 7)")

    with rasterio.open(path) as src:
        assert src.width == cfg.get("grid", "width")
        assert src.height == cfg.get("grid", "height")
        assert src.crs.to_string() == cfg.get("grid", "crs")
        assert src.dtypes[0] == "float32"
        nodata = src.nodata
        assert nodata == pytest.approx(cfg.get("inference", "nodata"))
        block = src.read(1, window=Window(0, src.height // 2, min(512, src.width), 8))

    valid = block[~np.isclose(block, nodata) & np.isfinite(block)]
    if valid.size:
        assert valid.min() >= 0.0 and valid.max() <= 1.0
