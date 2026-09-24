# Prediction API

FastAPI service around the trained bundles. It adds no modelling: every number
returned comes from a saved `ModelBundle`, the shared combiners, or a Stage 5
report.

**Entry point:** `api.app.main:app`

```bash
python -m uvicorn api.app.main:app --host 127.0.0.1 --port 8000
```

Interactive documentation: <http://127.0.0.1:8000/docs>. OpenAPI schema:
`/openapi.json`. When `frontend/dist` exists the built simulator is served at
`/`, so one process serves both.

---

## Conventions

**Errors** always take one shape, so clients branch on `code`, not prose:

```json
{ "error": { "code": "invalid_input", "message": "Some scenario values are invalid.",
             "fields": [{ "field": "slope", "message": "Input should be less than or equal to 90" }] } }
```

| Code | HTTP | Meaning |
|---|---|---|
| `models_loading` | 503 | Bundles still loading; retry (the UI polls `/api/health`). |
| `models_unavailable` | 503 | Artefacts missing or inconsistent; the message names the pipeline stage to run. |
| `invalid_input` | 422 | Validation failed; `fields` lists each offending factor. |
| `map_loading` | 503 | District thumbnail still rendering. |
| `layer_not_found` | 404 | Unknown district layer. |
| `internal_error` | 500 | Unexpected failure; details go to the API log, not the response. |

**Validation policy.** Physically impossible values are rejected (slope > 90°,
NDVI > 1, negative elevation, unknown class codes, unknown fields). Values
that are merely outside the *training* range are accepted and returned in
`warnings`, because refusing them would hide that the models are extrapolating.

**Responses are `application/json; charset=utf-8`** — the wording contains em
dashes and degree signs, and some clients otherwise decode them as Latin-1.

---

## Endpoints

### `GET /api/health`

Loading state. Answers immediately, including while models load.

```json
{ "status": "ready", "stage": "Ready", "error": null, "elapsed_s": 5.0,
  "models": ["topographic","hydrological","geological","anthropogenic","full"],
  "report_warnings": [], "district_map": "ready" }
```

`status`: `idle` | `loading` | `ready` | `error`.

### `GET /api/metadata`

Everything the UI needs to render controls and label results: the 14 factor
specifications (label, unit, theme, physical bounds, training percentiles,
categorical levels with sample counts and landslide rates), per-model spatial-CV
metrics read from the bundle sidecars, the operating threshold with its
out-of-fold metrics, the class bands, the planner decisions, the confidence
method, dataset counts and the standing caveats.

### `POST /api/predict`

```json
{ "factors": { "elevation": 506, "slope": 6.4, "aspect": 188, "plan_curv": 0.0,
               "profile_curv": 0.0, "TWI": 6.62, "SPI": 10.87, "dist_drainage": 491,
               "rainfall": 2773, "lithology": 1, "dist_fault": 193,
               "dist_road": 526, "LULC": 10, "NDVI": 0.688 },
  "explain": true }
```

All 14 factors are required. `lithology` ∈ {−1, 1, 2, 3, 4, 5}; `LULC` is an ESA
WorldCover code. `explain` defaults to `true`; set it `false` to skip SHAP
(roughly an order of magnitude faster).

Returns `prediction` (susceptibility, threshold, `above_threshold`, class,
`full_model`, `thematic_mean`, `disagreement`, `confidence`, `confidence_split`,
`decision`), `models[]` (calibrated and raw probability per model),
`explanations` (per-model SHAP in log-odds, summed to source factors),
`warnings[]`, `inputs`, `timing_ms` and `generated_utc`.

Confidence and `decision` are `null` when the Stage 5 reports are absent —
the API does not substitute a default.

### `POST /api/sensitivity`

```json
{ "factors": { "...": "as above" }, "factor": "dist_road", "points": 31 }
```

Re-scores the scenario across one factor, holding the rest fixed. Numeric
factors are swept across the 1st–99th training percentile, **spaced by
percentile** so points cluster where the data does; categorical factors sweep
every level seen in training. The scenario's current value is always included.
This is an individual conditional expectation curve for one scenario, not a
global effect. `points` ∈ [5, 81].

### `GET /api/samples/random?label=0|1`

One real row from the training data, with a note that the models have seen it,
so its score is not an independent test.

### `GET /api/district/meta`

Grid (CRS, bounds, thumbnail size), available layers, and the projected
coordinates of the 417 recorded landslides.

### `GET /api/district/image/{layer}.png`

`relief` (hillshade) and, once Stage 7 has run, `susceptibility`.

### `GET /api/district/sample?x=&y=`

Reads all 14 rasters at a coordinate in EPSG:32643 (UTM 43N) — the same values
district inference uses. Returns the factors, latitude/longitude, distance to
the nearest recorded landslide, and any `issues` (outside the grid, nodata, or
a value outside physical limits).

---

## Requirements

The API needs calibrated bundles in `outputs/models/`:

```bash
python run_pipeline.py stage 1 --to 5
```

Confidence and the planner decision additionally read
`outputs/reports/calibration_threshold.json` and `oof_calibrated.csv`. The
district susceptibility layer needs Stage 7. Without those, the service still
runs and says precisely what is missing.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `LANDSLIDE_DATA_ROOT` | from `config/config.local.yaml` | Data assets location |
| `LANDSLIDE_API_CORS_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173` | Browser origins allowed to call the API |

There are no credentials, keys or tokens in this service.
