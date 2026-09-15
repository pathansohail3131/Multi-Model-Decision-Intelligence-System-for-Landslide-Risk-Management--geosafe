# Landslide Susceptibility & Decision Intelligence — Uttara Kannada

District-wide landslide susceptibility mapping for Uttara Kannada, Karnataka,
with the decision layers a planner actually needs on top of it: a confidence
surface, physical interpretability, and an exposure-weighted prioritisation.

> **Build status: all 8 stages complete.** 20/20 tests passing.
> Headline: **ROC-AUC 0.8694 / AUC-PR 0.6564** under 5-fold spatial block CV —
> the honest figure, against 0.9132 / 0.7212 under a random split.
> See [`docs/FINAL_AUDIT.md`](docs/FINAL_AUDIT.md) for the full audit and
> [`docs/results.md`](docs/results.md) for every measured number.
>
> Two of the 14 conditioning-factor rasters could not be transferred during the
> build session, so the district rasters in `outputs/rasters/` are a
> provisional 2-of-4-theme run, suffixed `_partial_2of4themes`. Re-running
> `stage 7 --to 8` with all rasters present produces the full map with no code
> change.

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
python -m pytest tests/ -q
```

---

## Project layout

```
landslide/
├── config/
│   ├── config.yaml                 # tracked, portable
│   ├── config.local.yaml           # gitignored, machine-specific
│   └── config.local.yaml.example
├── docs/
│   ├── methodology.md              # every decision, with justification and rejected alternatives
│   ├── results.md                  # metrics, ablation, spatial-vs-random CV, SHAP
│   └── data_dictionary.md          # every column and layer: source, units, CRS, limitations
├── notebooks/                      # exploration only, never the pipeline
├── outputs/                        # generated; gitignored
│   ├── figures/  logs/  models/  rasters/  reports/
├── src/
│   ├── config.py                   # the only module that knows where anything lives
│   ├── data/                       # loading, validation, audit
│   ├── features/                   # encoding and thematic groups
│   ├── models/                     # estimators, combiners, persistence
│   ├── evaluation/                 # spatial CV, metrics, ablation
│   ├── inference/                  # windowed raster prediction
│   ├── decision/                   # rainfall, exposure, prioritisation
│   ├── visualisation/              # figures and maps
│   └── utils/                      # seeding, logging
├── tests/
├── requirements.txt
└── run_pipeline.py                 # single entry point
```

---

## Deliverables

- [x] Reproducible project scaffold, pinned dependencies, single entry point
- [x] District-wide susceptibility raster *(provisional — 2 of 4 themes)*
- [x] District-wide confidence raster *(provisional — 2 of 4 themes)*
- [x] Prioritised output combining susceptibility, confidence and exposure
- [x] SHAP interpretability, including cross-theme agreement
- [x] Full documentation set

---

## Documentation

| Document | Contents |
|---|---|
| [`docs/methodology.md`](docs/methodology.md) | Every methodological decision with its justification, including the options that were rejected and why. |
| [`docs/results.md`](docs/results.md) | Metrics tables, the ablation study, spatial vs random CV, SHAP findings. |
| [`docs/data_dictionary.md`](docs/data_dictionary.md) | Every column and raster layer: source, units, resolution, CRS, derivation, known limitations. |
| [`docs/FINAL_AUDIT.md`](docs/FINAL_AUDIT.md) | Final audit: what every code file does, what gives more accuracy (ranked by measured effect), the final system, and the honest weaknesses. |

A standing limitations section is maintained in `docs/methodology.md`. It is
not a formality — inventory completeness, temporal mismatch, resolution
constraints and unvalidated components are recorded there as they are
discovered, including the ones that weaken the result.
