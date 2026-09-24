# Landslide Susceptibility & Decision Intelligence — Uttara Kannada

District-wide landslide susceptibility mapping for Uttara Kannada, Karnataka,
with the decision layers a planner actually needs on top of it: a confidence
surface, physical interpretability, and an exposure-weighted prioritisation.

> **Build status: all 8 stages complete, on all 14 rasters.**
> Headline: **ROC-AUC 0.8694 / AUC-PR 0.6564** under 5-fold spatial block CV —
> the honest figure, against 0.9132 / 0.7212 under a random split.
> See [`docs/FINAL_AUDIT.md`](docs/FINAL_AUDIT.md) for the full audit and
> [`docs/results.md`](docs/results.md) for every measured number.
>
> The district rasters in `outputs/rasters/` are the **complete** four-theme run
> (2026-09-18): 10,928,670 valid pixels, **467 km² (4.6% of the district) above
> the 0.1374 operating threshold**, and a top-1% priority shortlist of 101 km².
> Planner quadrants: act now 37 km², survey first 430 km², monitor 4,640 km²,
> deprioritise 5,034 km². An earlier run produced a provisional 2-of-4-theme
> map; those `_partial_2of4themes` outputs are superseded, and §9 of
> `docs/results.md` still describes that earlier partial map.
>
> An [interactive 3D simulator](#interactive-simulator) runs these models on any
> scenario or any cell of the district.

---

## The problem

Uttara Kannada sits in the Western Ghats: steep, densely forested, and hit by
extreme monsoon rainfall. Landslides recur, cut the NH-66 corridor and the ghat
roads, and threaten settlements. District planners have limited survey and
mitigation budget and need to know where to spend it.

The **technical** problem is binary classification over terrain and
environmental conditions, rendered as a district raster.

The **decision** problem is larger, and it is the one this project is actually
built around. A single probability per pixel is not actionable. A planner needs
to know:

- how confident the system is,
- which physical factors drive a given prediction,
- and how susceptibility interacts with what is exposed there.

So the system outputs decision-relevant information, not just a probability
surface.

### Why this is a rebuild

A previous version trained five algorithms (Random Forest, XGBoost, CatBoost,
SVM, ANN) on identical features and stacked them. It did not work:

| Model | ROC-AUC |
|---|---|
| Stacked ensemble | 0.9225 |
| **CatBoost alone** | **0.9329** |

The stack lost to its own best member. The cause was structural rather than a
tuning problem: three of the five learners were tree ensembles trained on the
same columns, so their errors were correlated and averaging them added variance
without adding information, while the two genuinely different learners (SVM,
ANN) were too weak to contribute. The meta-learner had nothing to arbitrate.

This rebuild does not repeat that design. See
[Architecture](#architecture-the-three-senses-of-multi-model).

---

## Architecture: the three senses of "multi-model"

### (a) Diversity from data views, not algorithms

One strong learner is trained on each of four **thematic feature subsets**:

| Theme | Conditioning factors |
|---|---|
| Topographic | elevation, slope, aspect, plan curvature, profile curvature |
| Hydrological | TWI, SPI, distance to drainage, rainfall (5-yr average) |
| Geological | lithology, distance to fault |
| Anthropogenic / surface | distance to road, LULC, NDVI |

Each model is individually strong and genuinely blind to what the others know,
which produces real error decorrelation rather than assumed diversity. The
themes partition the 14 factors exactly — no factor appears in two themes, and
a test enforces it, because an overlap would reintroduce precisely the shared
input that made the old stack's members agree for the wrong reason.

It is also physically interpretable. "Flagged by terrain but not by geology" is
a statement a planner can act on; "the ensemble said 0.71" is not.

A fifth **full-feature model** is trained as the performance benchmark. It is
not a thematic member and never contributes to the confidence calculation.

### (b) Multi-model for uncertainty, not just accuracy

The ensemble is not asked to beat the best single model on AUC. It is asked
something a single model structurally cannot answer. Two rasters are produced:

- **Susceptibility** — the aggregated prediction.
- **Confidence** — derived from disagreement (variance) across the thematic
  models.

| | High agreement | High disagreement |
|---|---|---|
| **High susceptibility** | Act. | Send a survey team before committing resources. |
| **Low susceptibility** | Deprioritise. | Do not clear it yet. |

That distinction is the product. A consequence worth stating plainly: **if the
thematic models never disagree, the confidence raster is worthless.** An
ablation showing the ensemble gives up a little AUC to the full-feature model
is therefore not automatically a failure — it may be the price of the
confidence layer. Whether that price is worth paying is reported from measured
numbers in `docs/results.md`, not assumed here.

### (c) Multi-model across tasks, not just one task

Models that do different jobs, chained — rather than a committee voting on one
job:

```
Susceptibility model (ML, static)
        |
        v
Rainfall threshold component (intensity-duration)
        |
        v
Exposure overlay (roads, built-up land cover)
        |
        v
Prioritised output for planners
```

A ground-deformation layer (NISAR L-band InSAR) can be slotted in later as a
**post-hoc activity multiplier with no retraining**. It is deliberately not a
training feature: NISAR observations postdate the landslide inventory, so using
them as a predictor would introduce temporal leakage and force a full retrain
on every new acquisition. As a multiplier applied after inference, it is a
configuration change (`decision.nisar` in `config/config.yaml`).

---

## Data

All data assets are **existing, validated inputs**. This project does not
regenerate them.

| Asset | Contents |
|---|---|
| `data/final/training_dataset_final.csv` | 4,940 labelled samples — 417 landslide points (Geological Survey of India records, cross-verified and deduplicated) and 4,523 non-landslide points from slope-stratified sampling. Class ratio 1:10.8. |
| `data/processed/*.tif` | Aligned rasters for all 14 conditioning factors, co-registered to a common grid, CRS and extent. |
| `data/raw/boundary/` | District boundary shapefile. |
| `data/samples/` | Landslide inventory shapefile and verified point CSV. |

**Grid** (verified, not assumed): 3738 × 5554 px at 30.4625 m, EPSG:32643
(WGS 84 / UTM 43N). 10,929,202 in-district pixels ≈ 10,142 km², against the
district's official 10,291 km².

Full column-by-column and layer-by-layer detail, including known limitations,
is in [`docs/data_dictionary.md`](docs/data_dictionary.md).

### Data findings you should know about

These emerged from auditing the delivered assets rather than taking them at
face value, and each one changed a design decision. They are reproduced by
`python run_pipeline.py stage 1` and documented in full in
[`docs/methodology.md`](docs/methodology.md).

1. **`lithology = -1` on 157 samples** (43 of them positives) is not random
   missingness. Those points form a coastal strip — mean elevation 45 m against
   414 m district-wide — where GLiM polygon coverage is absent. Their landslide
   rate is 27.4% against an 8.4% baseline, so the code is spatially confounded
   with the outcome. Kept as an explicit `unmapped_coastal_gap` category and
   disclosed, rather than dropped (which would delete 10% of positives and the
   entire coastal hazard belt) or imputed (which would invent geology).
2. **Every `slope == 0` cell carries `aspect = 270.0` exactly.** That is the
   terrain software's flat-cell convention, not a real direction. Encoded
   naively it would inject a fake due-west signal into flat terrain, so flat
   cells are flagged and their aspect components zeroed.
3. **`taluk` and `slide_name` are non-null only on positive samples.** Either
   one is a perfect label leak. Both are listed in
   `features.forbidden_columns` and a test enforces it.
4. **`SPI` spans four orders of magnitude** (max 2,250,643; skew 46.3).
   `log1p` brings skew to 1.29.
5. **Nodata differs per raster** — −9999 on the GDAL-derived layers, −32768 on
   the WhiteboxTools-derived hydrological layers. Read per file; assuming one
   value would silently write garbage into the map.
6. **Spatial clustering is real.** 86 of 417 positives sit within 100 m of
   another positive (minimum separation 0.65 m). Negatives were sampled with a
   ≥101 m buffer from any positive. This is why evaluation is spatially
   blocked, and the buffer itself is recorded as a limitation.

---

## Installation

Requires Python 3.12.

```bash
git clone <this-repo>
cd landslide
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Every dependency is pinned exactly and the set was resolved together. The
previous version shipped no requirements file at all, and its saved models
became unloadable once the local scikit-learn and CatBoost versions moved on.

**The pins are load-bearing, not hygiene.** One measured example: with
`xgboost` 3.4.1 instead of the pinned 3.2.0, Stage 4 aborts partway through the
mixed-learner comparison. The newer release refuses to predict on a categorical
level that a fold's training split never saw, and spatial blocking guarantees
such folds — `herbaceous_wetland` has a single sample in the whole dataset.
Verify with `pip install -r requirements.txt` before reporting a stage failure.

### Point the project at your data

The data assets are large and live outside the repository. There is exactly one
machine-specific setting:

```bash
cp config/config.local.yaml.example config/config.local.yaml
# edit config/config.local.yaml and set paths.data_root
```

`config/config.local.yaml` is gitignored and deep-merged over `config.yaml`.
For a one-off run you can instead set an environment variable:

```bash
export LANDSLIDE_DATA_ROOT=/path/to/data     # Windows: set LANDSLIDE_DATA_ROOT=...
```

Precedence: environment variable → `config.local.yaml` → `config.yaml`.

No module under `src/` contains an absolute path, and
`tests/test_config.py::test_no_absolute_paths_in_source` fails the build if one
appears.

### Verify the setup

```bash
python run_pipeline.py check
```

This reports the resolved paths, every installed dependency version, and every
missing data asset **at once** — rather than stopping at the first problem and
making you rerun after each fix.

---

## Running the pipeline

There is a single entry point. No notebook needs to be run first; notebooks
under `notebooks/` are exploration only and are never imported by the pipeline.

```bash
python run_pipeline.py check        # validate config, data and dependencies
python run_pipeline.py stages       # list stages and their status
python run_pipeline.py all          # run everything end to end
python run_pipeline.py stage 3      # run one stage
python run_pipeline.py stage 3 --to 5   # run a range
```

| Stage | Name | What it does |
|---|---|---|
| 1 | `data-audit` | Load and validate assets; produce the reproducible data audit. |
| 2 | `features` | Circular aspect encoding, flat flag, `log1p` SPI, categoricals, thematic groups. |
| 3 | `spatial-cv` | Spatial block CV vs random CV; empirical class-imbalance comparison. |
| 4 | `models-ablation` | Four thematic models + full-feature benchmark; combination-rule ablation. |
| 5 | `confidence-threshold` | Calibration, disagreement confidence, operating threshold. |
| 6 | `interpretability` | SHAP per thematic model; cross-theme agreement. |
| 7 | `raster-inference` | Windowed district-wide susceptibility and confidence rasters. |
| 8 | `decision-layer` | Rainfall thresholds, exposure overlay, prioritised output, NISAR slot. |

Each run writes a timestamped log to `outputs/logs/`, recording the resolved
configuration, the data root and the determinism settings in force, so any
number in `docs/results.md` can be traced back to the run that produced it.

### Reproducibility

- One seed (`project.random_seed`) propagates to Python, NumPy and every model.
  Estimators receive `random_state` explicitly rather than relying on a global
  RNG, which is how "reproducible" pipelines quietly stop being reproducible.
- Stages whose numeric output is reported (1–6) run with thread pools pinned to
  1, removing parallel floating-point reduction nondeterminism. This costs
  speed and is worth it: the headline ablation gap is ~0.01 AUC, which is
  within the range run-to-run thread variation can produce.
- Stages 7–8 relax that, since their output is a smooth probability surface
  where a last-bit difference is immaterial.
- Every model is serialised **with its preprocessing attached**. A model that
  can be loaded without its transformer will eventually be used without one,
  and the resulting predictions look entirely plausible while being wrong.

### Tests

```bash
python -m pytest tests -q
python -m compileall -q api src tests scripts run_pipeline.py
```

Tests that need trained bundles or rasters are **skipped, not failed**, when
those artefacts are absent, so a clean checkout still validates configuration,
imports, the ensemble rules and the class logic. Run the pipeline through
Stage 5 to enable the model and API tests.

The load-bearing one is
`test_api.py::test_predict_matches_the_bundles_directly`: it recomputes the
production score from the saved bundles and asserts the HTTP response matches
to 1e-6, so a second scoring path can never be introduced silently. What each
file covers is in [`docs/testing.md`](docs/testing.md).

### Batch scoring

To score a CSV of sites without starting the API:

```bash
python -m scripts.predict --template 3 > sites.csv     # column template
python -m scripts.predict sites.csv --output scored.csv
python -m scripts.predict coords.csv --from-rasters --x-column X --y-column Y
```

`--from-rasters` reads the 14 factors from the district rasters at each row's
UTM 43N coordinates. Output adds the per-model probabilities, susceptibility,
its class, the threshold flag, disagreement and — when Stage 5 reports are
present — confidence and the planner decision.

---

## Interactive simulator

A browser-based 3D simulator for the trained models. You configure one 30 m
terrain cell, press **Run simulation**, and the real bundles score it; the site
model and the analysis panel then show the result.

```
React + three.js frontend  ──POST /api/predict──▶  FastAPI (api/)
   controls, 3D site model                          ModelBundle.predict_proba
   analysis panels          ◀──prediction JSON───   (encoder → CatBoost → calibrator)
                                                     full_plus_thematic_mean, disagreement,
                                                     confidence, SHAP, planner quadrant
```

### Requirements

- Python environment from [Installation](#installation), plus
  `pip install -r requirements-api.txt`
- Node.js 20 or later (tested with 24)
- Trained, calibrated bundles in `outputs/models/`: run the pipeline through
  Stage 5 first (`python run_pipeline.py stage 1 --to 5`). Confidence and the
  planner decision also read `outputs/reports/calibration_threshold.json` and
  `oof_calibrated.csv` from Stage 5.

### Start it

Backend, from the project root:

```bash
python -m uvicorn api.app.main:app --host 127.0.0.1 --port 8000
```

Frontend, in a second terminal:

```bash
cd frontend
npm install
npm run dev          # http://localhost:5173, proxies /api to port 8000
```

For a single-process demo, build the frontend once; the API then serves it at
http://127.0.0.1:8000:

```bash
cd frontend && npm run build
```

| Variable | Where | Purpose |
|---|---|---|
| `LANDSLIDE_DATA_ROOT` | backend | Overrides `paths.data_root`, as for the pipeline |
| `VITE_API_TARGET` | `npm run dev` | Backend URL for the dev proxy (default `http://127.0.0.1:8000`) |
| `VITE_API_BASE` | `npm run build` | API origin if the frontend is hosted separately (default: same origin) |

### Prediction API

| Method | Path | Returns |
|---|---|---|
| GET | `/api/health` | Loading status of the bundles — the startup screen reads this |
| GET | `/api/metadata` | Factor specs with training-data ranges, per-model spatial-CV metrics, threshold, classes, caveats |
| POST | `/api/predict` | `{"factors": {…14 raw factors…}}` → per-model calibrated and raw probabilities, production score, class, threshold flag, disagreement, confidence, planner decision, per-row SHAP, input warnings |
| POST | `/api/sensitivity` | Re-scores the scenario across one factor's training distribution |
| GET | `/api/samples/random?label=0\|1` | A real training row (flagged as seen by the model) |
| GET | `/api/district/meta`, `/api/district/image/{layer}.png` | District map thumbnail and the recorded-landslide inventory |
| GET | `/api/district/sample?x=&y=` | The 14 factors read from the rasters at a UTM 43N coordinate |

Interactive docs are at http://127.0.0.1:8000/docs. Inputs are validated
against physical limits (HTTP 422 with per-field messages). Values outside the
*training* range are accepted but returned as warnings, because the models are
extrapolating there.

### How the scene corresponds to the model

The 3D block is a schematic of the assessed cell and the 1.2 km of hillslope
around it, generated from the 14 inputs. **It is a visual aid: only the
parameter values are sent to the model.**

- **Drawn to scale:** slope angle (no vertical exaggeration), aspect (the
  compass turns so downslope faces the viewer), elevation (sea level floods the
  block near the coast), and distances to stream, road and lineament.
- **Positions are schematic:** the rasters give distances, not directions, so
  stream, road and lineament are placed at the right distance but an arbitrary
  bearing.
- **Qualitative:** curvature, wetness (TWI), gully depth (SPI), rainfall
  density, vegetation density (NDVI) and rock pattern, each normalised against
  the real training distribution.
- **Model lens:** dims everything except what one thematic model can see. This
  is the architecture made visible: each thematic model is blind to the others'
  factors, and their disagreement is the confidence signal.
- **Result:** the halo and flag take the susceptibility class and planner
  decision. A hatched outline appears downslope when the cell is flagged; it is
  labelled illustrative because the model predicts susceptibility, not failure
  extent.

Every number in the analysis panel comes from the API. The pipeline strip
animates while the request runs, but its timing is illustrative. The
**What-if sweep** re-runs the real models across one factor. **Scenario runs**
compares completed runs against a pinned baseline.

The simulator falls back to a 2D cross-section of the same terrain function if
WebGL is unavailable, and respects `prefers-reduced-motion`.

---

## Project layout

```
landslide/
├── api/                            # FastAPI prediction service
│   ├── app/
│   │   ├── main.py                 # app factory, CORS, lifespan, serves frontend/dist
│   │   ├── dependencies.py         # process-wide service singletons
│   │   ├── core/                   # HTTP settings; error types and handlers
│   │   ├── routers/                # health, prediction, district
│   │   ├── schemas/                # request models; conditioning-factor catalogue
│   │   └── services/               # the only modules that touch model bundles
│   └── README.md
├── frontend/                       # React + TypeScript + three.js simulator
│   ├── src/
│   │   ├── three/                  # scene: terrain model and shader, land cover, camera
│   │   ├── components/             # controls, results, district locator, viewport HUD
│   │   ├── store/  hooks/  services/  utils/  types/  styles/
│   └── README.md
├── src/                            # the ML pipeline
│   ├── config.py                   # the only module that knows where anything lives
│   ├── data/                       # loading, validation, audit, raster point sampling
│   ├── features/                   # encoding and thematic groups
│   ├── models/                     # estimators, bundles, calibration, ensemble combiners
│   ├── evaluation/                 # spatial CV, metrics, ablation, SHAP interpretability
│   ├── inference/                  # windowed district raster prediction
│   ├── decision/                   # rainfall, exposure, prioritisation
│   ├── visualisation/              # figures and maps
│   └── utils/                      # seeding, logging
├── scripts/
│   └── predict.py                  # batch CSV scoring without starting the API
├── tests/                          # conftest + config, models, api, validation, inference
├── config/
│   ├── config.yaml                 # tracked, portable
│   ├── config.local.yaml           # gitignored, machine-specific
│   ├── config.local.yaml.example
│   └── README.md
├── docs/
│   ├── architecture.md             # how the parts fit together, and where each concept lives
│   ├── api.md                      # endpoint reference
│   ├── testing.md                  # what each test file covers
│   ├── methodology.md              # every decision, with justification and rejected alternatives
│   ├── results.md                  # metrics, ablation, spatial-vs-random CV, SHAP
│   ├── data_dictionary.md          # every column and layer: source, units, CRS, limitations
│   └── FINAL_AUDIT.md
├── notebooks/                      # exploration only, never the pipeline
├── outputs/                        # generated; gitignored
│   └── figures/  logs/  models/  rasters/  reports/
├── .github/workflows/ci.yml        # compile, import, tests, frontend build
├── .env.example                    # optional environment variables; no secrets
├── pyproject.toml                  # project metadata and pytest configuration
├── requirements.txt                # pipeline dependencies, exactly pinned
├── requirements-api.txt            # HTTP service only
└── run_pipeline.py                 # single entry point for the 8 stages
```

Training and evaluation have no separate scripts: they are pipeline stages
(`run_pipeline.py stage 4`, `stage 3`, `stage 6`). A second code path for
numbers that must stay reproducible would be a liability, not a convenience.

---

## Deliverables

- [x] Reproducible project scaffold, pinned dependencies, single entry point
- [x] District-wide susceptibility raster *(complete — all 4 themes)*
- [x] District-wide confidence raster *(complete — all 4 themes)*
- [x] Prioritised output combining susceptibility, confidence and exposure
- [x] SHAP interpretability, including cross-theme agreement
- [x] Interactive 3D prediction simulator over the trained models
- [x] Full documentation set

---

## Documentation

| Document | Contents |
|---|---|
| [`docs/architecture.md`](docs/architecture.md) | How the pipeline, API, simulator and batch script fit together, and where every concept lives in the tree. |
| [`docs/api.md`](docs/api.md) | Endpoint reference: payloads, error codes, validation policy, configuration. |
| [`docs/testing.md`](docs/testing.md) | What each test file covers, and why some tests skip. |
| [`docs/methodology.md`](docs/methodology.md) | Every methodological decision with its justification, including the options that were rejected and why. |
| [`docs/results.md`](docs/results.md) | Metrics tables, the ablation study, spatial vs random CV, SHAP findings. |
| [`docs/data_dictionary.md`](docs/data_dictionary.md) | Every column and raster layer: source, units, resolution, CRS, derivation, known limitations. |
| [`docs/FINAL_AUDIT.md`](docs/FINAL_AUDIT.md) | Final audit: what every code file does, what gives more accuracy (ranked by measured effect), the final system, and the honest weaknesses. |

A standing limitations section is maintained in `docs/methodology.md`. It is
not a formality — inventory completeness, temporal mismatch, resolution
constraints and unvalidated components are recorded there as they are
discovered, including the ones that weaken the result.
