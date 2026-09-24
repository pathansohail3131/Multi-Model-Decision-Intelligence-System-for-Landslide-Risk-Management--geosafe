# Testing

```bash
python -m pytest tests -q          # whole suite
python -m pytest tests/test_models.py -v
python -m compileall -q api src tests scripts run_pipeline.py
```

Configuration lives in `pyproject.toml` (`[tool.pytest.ini_options]`).

---

## What each file covers

| File | Covers |
|---|---|
| `conftest.py` | Shared session fixtures: config, loaded bundles, training data, a real landslide row, and a `TestClient` that waits for the models to load. Also the `requires_bundles` / `requires_rasters` skip markers. |
| `test_config.py` | Configuration loading and merge precedence; that no absolute path appears under `src/`; that leaky columns stay in `forbidden_columns`; that the thematic partition is exact. |
| `test_models.py` | Bundles carry encoder, calibrator and threshold; all bundles share one threshold; the themes partition the factors; predictions are finite probabilities of the right shape; column order does not change a score; encoding and prediction are repeatable; binary prediction uses the derived threshold; the production blend and the disagreement/confidence rules behave as documented. |
| `test_api.py` | Health, OpenAPI route list, metadata completeness, **the API's score equals a direct bundle computation**, response shape, SHAP summing to the model output, `explain=false`, sensitivity sweeps, district metadata and sampling, unknown layer handling. |
| `test_validation.py` | Missing fields, wrong types, physically impossible values, NaN, unknown class codes, unexpected fields, boundary values accepted, out-of-training-range flagged rather than rejected, flat-cell and sparse-level warnings, sensitivity point bounds. |
| `test_inference.py` | Class bands tile [0, 1] with the threshold as a boundary; class assignment at boundaries; all themes usable when rasters are present; the geospatial path (sample rasters at coordinates → score); the produced district GeoTIFF matches the configured grid, dtype, nodata and value range. |

## Skipping is deliberate

Model bundles and rasters are large and live outside the repository. Tests that
need them are **skipped, not failed**, when they are absent, so a clean checkout
and CI still validate configuration, imports, class logic and the ensemble
rules. Run the pipeline through Stage 5 to enable the model and API tests, and
Stage 7 for the district raster check.

## The one test that reaches the whole stack

`test_api.py::test_predict_matches_the_bundles_directly` re-computes the
production score with `ModelBundle.predict_proba` and
`full_plus_thematic_mean`, then asserts the HTTP response matches to 1e-6. If
anyone introduces a second scoring path in the API, that test fails.

## Manual verification not covered by pytest

The browser simulator (3D scene, camera, lens, district picker, mobile layout,
WebGL fallback) is verified by driving a real browser; it has no automated
suite in this repository. See the end-to-end checklist in the README.
