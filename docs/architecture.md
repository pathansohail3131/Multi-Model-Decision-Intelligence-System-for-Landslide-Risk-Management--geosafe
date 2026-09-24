# Architecture

How the parts fit together, and where each concept lives in the tree. The
*reasoning* behind the modelling decisions is in
[`methodology.md`](methodology.md); this document is the map.

---

## 1. System overview

```
                 ┌──────────────────────────────────────────┐
  data/          │  run_pipeline.py — 8 stages, one entry   │
  (outside repo) │  point, writes outputs/ and docs numbers │
      │          └───────────────┬──────────────────────────┘
      │                          │ trains, calibrates, writes
      ▼                          ▼
  14 conditioning        outputs/models/*.joblib   outputs/rasters/*.tif
  factor rasters         (bundle = encoder +       outputs/reports/*.json
  + training CSV          estimator + calibrator
                          + threshold)
                               │
             ┌─────────────────┴──────────────────┐
             ▼                                    ▼
   api/ (FastAPI)                        scripts/predict.py
   loads bundles once, scores            batch CSV scoring,
   one cell per request                  same bundles, no server
             │
             ▼
   frontend/ (React + three.js)
   3D simulator, one cell at a time
```

Three consumers of the trained models — the pipeline's own raster stage, the
HTTP API, and the batch script — all score through the same
`ModelBundle.predict_proba` and the same combiners. There is no second
implementation of the scoring rule anywhere in the tree.

---

## 2. The ML pipeline (`src/`, driven by `run_pipeline.py`)

| Concern | Module | Notes |
|---|---|---|
| Configuration | `src/config.py` | The only module that knows where anything lives. No absolute paths anywhere under `src/` (enforced by a test). |
| Data loading, audit, validation | `src/data/` | `loaders.py` also holds the **geospatial sampling** helpers (`sample_raster_at_points`, `sample_factors_at_points`); `validation.py` asserts grid, CRS and nodata per file. |
| Feature encoding | `src/features/` | `encoders.py` (circular aspect, flat flag, `log1p` SPI, categorical labels), `themes.py` (the thematic partition), `build.py`. |
| Estimators and persistence | `src/models/` | `estimators.py` (CatBoost / XGBoost / logistic + imbalance strategies), `bundle.py` (**model + preprocessing + calibrator + threshold as one artefact**), `train.py`, `calibration.py`. |
| **Ensemble** | `src/models/combiners.py` | `full_plus_thematic_mean` (the production rule), `disagreement`, `confidence_from_disagreement`, plus the alternative rules compared in the Stage 4 ablation. |
| Evaluation | `src/evaluation/` | `spatial_cv.py` (block CV), `metrics.py`, `runner.py`, `spatial_stage.py`. |
| **Explainability** | `src/evaluation/interpretability.py` | CatBoost native SHAP (`get_feature_importance(type="ShapValues")`), per-model importance and cross-theme agreement. |
| **Geospatial inference** | `src/inference/raster.py` | Windowed district prediction, `susceptibility_classes`, confidence scaling. |
| Decision layers | `src/decision/layers.py` | Rainfall intensity–duration component, exposure overlay, priority surface, `QUADRANT_INTERPRETATION`. |
| Figures | `src/visualisation/maps.py` | |
| Seeding and logging | `src/utils/` | |

`src/` keeps the layout the methodology and audit documents reference. The
ensemble, explainability and geospatial concerns named in the table are real
and located above; they are modules rather than separate top-level packages
because each is one focused file inside the package that owns its data.

### Stage order

1 data-audit → 2 features → 3 spatial-cv → 4 models-ablation → 5
confidence-threshold → 6 interpretability → 7 raster-inference → 8
decision-layer. Stages 1–6 run with thread pools pinned to 1 for bit-level
reproducibility; 7–8 do not, because their output is a smooth surface.

---

## 3. The prediction API (`api/`)

```
api/
└── app/
    ├── main.py            FastAPI factory, CORS, lifespan, static mount
    ├── dependencies.py    process-wide service singletons + FastAPI deps
    ├── core/
    │   ├── config.py      HTTP settings (CORS origins, frontend dist path)
    │   └── errors.py      ServiceError/ServiceNotReady, one error body shape
    ├── routers/
    │   ├── health.py      /api/health
    │   ├── prediction.py  /api/metadata, /predict, /sensitivity, /samples/random
    │   └── district.py    /api/district/meta, /image/{layer}.png, /sample
    ├── schemas/
    │   ├── factors.py     the 14 factors: labels, units, physical bounds
    │   └── requests.py    pydantic request models
    └── services/
        ├── prediction.py  PredictionService: loads bundles, scores, explains
        └── district.py    DistrictService: map thumbnails, raster sampling
```

**Loading is asynchronous by design.** `lifespan` starts the model load on a
background thread and returns immediately, so `/api/health` can report honest
progress (`loading` → `ready`) instead of the server appearing hung for the
seconds it takes to unpickle five CatBoost models. Requests that arrive early
raise `ServiceNotReady` → HTTP 503 with a machine-readable code.

**The services are singletons** held in `dependencies.py`: they own the loaded
bundles, and rebuilding them per request would reload every model.

Endpoint details are in [`api.md`](api.md).

---

## 4. The simulator (`frontend/`)

React 19 + TypeScript + three.js (react-three-fiber), built by Vite.

| Layer | Location | Responsibility |
|---|---|---|
| API access | `src/services/predictionApi.ts` | The only module that calls `fetch`; maps failures to typed `ApiError`s and guards response shapes. |
| State | `src/store/simulationStore.ts` | Zustand store: phase, run step, inputs, scenario source, run history, lens, camera. |
| Run sequence | `src/hooks/useRunSimulation.ts` | validate → lock → prepare → **real inference** → result → analysis. |
| 3D scene | `src/three/` | `terrain/terrainModel.ts` is pure geometry (no three.js) and is shared with the 2D fallback; `SceneRoot.tsx` animates scene parameters in a mutable ref so React does not re-render per frame. |
| Panels | `src/components/` | controls, results, district locator, viewport HUD. |

No probability, metric or explanation is computed in the browser. Every number
displayed comes from an API response.

---

## 5. Data flow of one prediction

```
14 factor values (UI control, district map click, or training row)
   → POST /api/predict           validated against physical bounds
   → PredictionService.predict
       → bundle.encoder.transform (aspect sin/cos, is_flat, log1p SPI, categories)
       → 5 × CatBoost predict_proba          → raw probabilities
       → isotonic calibrator per model       → calibrated probabilities
       → full_plus_thematic_mean             → susceptibility
       → disagreement(std) → confidence_from_disagreement (Stage 5 ceiling)
       → susceptibility_classes → class; threshold → flag
       → quadrant (susceptibility × confidence) → planner decision
       → CatBoost ShapValues                 → per-factor contributions
   → JSON → 3D scene state + analysis panel
```

---

## 6. Outputs

| Path | Produced by | Contents |
|---|---|---|
| `outputs/models/*.joblib`, `*.json` | Stage 4–5 | Model bundles and readable provenance sidecars. |
| `outputs/rasters/*.tif` | Stage 7–8 | susceptibility, disagreement, confidence, exposure, priority — float32, `-9999` nodata, on the configured grid. |
| `outputs/reports/*.json`, `*.csv` | Stages 1–8 | Audit, CV, ablation, calibration, interpretability, inference and decision reports; every number quoted in the docs traces to one of these. |
| `outputs/figures/*.png` | Stage 8 | Maps and comparison figures. |
| `outputs/logs/*.log` | every run | Timestamped run log with the resolved configuration. |

`outputs/` is gitignored apart from `.gitkeep`: it is reproducible from code,
configuration and the data assets.
