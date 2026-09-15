# Final Audit — Landslide Risk Decision Intelligence System

Uttara Kannada district, Karnataka. Complete rebuild of the modelling and
application layer, built from scratch in eight stages.

**Audit date:** 2026-09-13 · **Seed:** 42 · **Status:** all 8 stages executed,
20/20 tests passing, 2 of 14 rasters pending transfer (§7).

---

## 1. Executive summary

| | |
|---|---|
| **Honest headline performance** | **ROC-AUC 0.8694, AUC-PR 0.6564** under 5-fold spatial block CV at 10 km |
| Previous version's reported figure | 0.9329 ROC-AUC — under a **random** split |
| Measured leakage in that comparison | **+0.0438 ROC-AUC, +0.0648 AUC-PR** |
| Adopted architecture | 4 thematic CatBoost models + full-feature benchmark, blended `0.5·(full + mean(thematic))` |
| Cross-theme error correlation | **0.181** — the design's core claim, confirmed |
| Cost of the confidence layer | **0.0024 AUC-PR** (previous stack lost 0.0104 for nothing) |
| Operating threshold | **0.1374**, cost-minimising at 10:1. Recall 0.650 vs **0.338** at the 0.5 default |
| District output | 10,929,202 px; **768.9 km² (7.6%)** above threshold; top-1% priority shortlist **102.5 km²** |
| Codebase | 7,544 lines Python, 2,289 lines docs/config, 20 tests |

**The system works and its central design claim is supported by measurement.**
Its largest unresolved weakness is that the most predictive single theme is also
the most epistemically compromised (§6).

---

## 2. What is in every code file

**7,544 lines of Python across 34 files.** No notebook is part of the pipeline.

### Entry point

| File | Lines | What it does |
|---|---|---|
| `run_pipeline.py` | 432 | The only way to run anything. Registers all 8 stages, their dependencies, and whether each needs the rasters. Sub-commands: `check`, `stages`, `stage N [--to M]`, `all`, `figures`. Sets determinism per stage (strict for 1–6 where numbers are reported, relaxed for 7–8 where output is a smooth surface). |

### Configuration and utilities

| File | Lines | What it does |
|---|---|---|
| `src/config.py` | 480 | The only module that knows where anything lives. Deep-merges `config.yaml` ← `config.local.yaml` ← `LANDSLIDE_DATA_ROOT`. Resolves every path; validates data availability and reports *all* problems at once rather than one per run. Distinguishes raster-needing stages from CSV-only ones. |
| `src/utils/seeds.py` | 109 | Seeds Python, NumPy and every estimator explicitly (never via a global RNG). `strict=True` pins BLAS/OpenMP threads to 1 to remove parallel floating-point reduction nondeterminism — necessary because the headline ablation gap is ~0.01 AUC. |
| `src/utils/logging_setup.py` | 118 | Console + timestamped file logging to `outputs/logs/`. Records resolved config, data root and determinism settings so any published number traces to one run. |

### Data layer — Stage 1

| File | Lines | What it does |
|---|---|---|
| `src/data/loaders.py` | 327 | Loads the CSV; drops the 4 physically impossible negative-elevation rows with `sample_id` logged. Reads raster metadata without loading pixels. Samples rasters at point coordinates using **each file's own nodata**, never a global constant. |
| `src/data/validation.py` | 244 | Asserts all 14 rasters share one grid, CRS and transform, and that this matches `config.yaml`. Computes the district footprint as an end-to-end CRS sanity check (10,142 km² vs official 10,291). Records the two nodata conventions in use. Degrades to a partial report rather than crashing on a missing layer. |
| `src/data/audit.py` | 570 | Turns every documented data claim into a reproducible script: class balance, completeness, label-leakage proof, categorical class rates, the lithology coverage gap, the flat-cell aspect artefact, skew, spatial clustering, trivial negatives, and the **empirical variogram** that measures the target's autocorrelation range. Also resolves which of the two rainfall rasters the CSV was actually built from. |

### Features — Stage 2

| File | Lines | What it does |
|---|---|---|
| `src/features/encoders.py` | 297 | `FeatureEncoder`, a fitted sklearn transformer that travels inside every model bundle. Four encodings: circular aspect → sin/cos; **flat-cell handling** (zero slope ⇒ zero aspect components + `is_flat` flag, so the software's literal 270.0 placeholder is never read as a westward bearing); `log1p(SPI)`; nominal codes → readable string categories. Enforces exact output column order at transform time. |
| `src/features/themes.py` | 181 | Defines and validates the thematic partition. `validate_partition` proves the 4 themes tile the 14 factors exactly once each — an overlap would reintroduce the shared input that made the old stack's members agree for the wrong reason. |
| `src/features/build.py` | 203 | `prepare_dataset` — the **single** function every later stage calls to get the encoded data, so Stage 4 and Stage 7 cannot diverge in how they encode. Persists the fitted encoder and a readable CSV of exactly what the models saw. |

### Models — Stages 4, 5

| File | Lines | What it does |
|---|---|---|
| `src/models/estimators.py` | 305 | Builds CatBoost / XGBoost / logistic estimators, wires the imbalance strategy, and coerces categoricals to whatever each learner needs (strings / pandas category / one-hot) in one place. `apply_imbalance_strategy` runs SMOTE-NC **inside the fold only**. |
| `src/models/bundle.py` | 231 | `ModelBundle` — the only unit of persistence. Carries estimator + fitted encoder + exact feature list + categoricals + calibrator + threshold + provenance. `predict_proba` takes **raw factors**, not encoded features, so a model physically cannot be applied to something its own encoder did not produce. `predict` refuses to run without a derived threshold. Writes a JSON sidecar readable without unpickling. |
| `src/models/combiners.py` | 253 | Six combination rules (mean, skill-weighted, rank-average, noisy-OR, max, plus the meta-learner in `train.py`) and the `disagreement` / `confidence_from_disagreement` functions that produce the project's headline output. |
| `src/models/train.py` | 460 | Stage 4. Trains 4 thematic + 1 benchmark on one shared fold structure; computes the error-decorrelation matrix; runs the full combination ablation; runs the per-theme learner comparison **and** the mixed-learner ensemble test; fits final models; persists bundles and out-of-fold predictions. |
| `src/models/calibration.py` | 400 | Stage 5. Isotonic calibration fitted on out-of-fold predictions and evaluated through a **second layer** of spatial folds. Quantifies how much apparent disagreement was calibration mismatch. Derives the operating threshold from an explicit cost ratio with sensitivity analysis, builds the four-quadrant decision table, and attaches calibrators + threshold back onto the saved bundles. |

### Evaluation — Stages 3, 6

| File | Lines | What it does |
|---|---|---|
| `src/evaluation/metrics.py` | 232 | The metric suite, AUC-PR first. Includes `flagged_rate`, because a threshold that flags 75% of the district is useless whatever its recall. `cost_minimising_threshold` and `threshold_candidates` derive four alternative operating points. |
| `src/evaluation/spatial_cv.py` | 263 | `SpatialBlockCV` — tiles the district into square blocks and assigns **whole blocks** to folds, greedily balanced by positive count so no fold is starved. `summarise_split` reports the geometry (blocks, points/block, test→train distances) that makes the leakage visible without running a model. |
| `src/evaluation/runner.py` | 241 | Executes CV and collects out-of-fold predictions — the backbone that lets a combination rule or calibrator be fitted honestly rather than on predictions the base models had memorised. |
| `src/evaluation/spatial_stage.py` | 218 | Stage 3. Sweeps block size, runs random CV alongside, quantifies the inflation, compares four imbalance strategies, and writes both selections back for Stage 4 to consume. |
| `src/evaluation/interpretability.py` | 304 | Stage 6. SHAP per model via CatBoost's native implementation (the generic tree explainer does not model CatBoost's ordered target statistics). Computes cross-theme factor agreement and analyses what drives the highest-disagreement pixels — i.e. what a survey team should go and check. |

### Inference and decision — Stages 7, 8

| File | Lines | What it does |
|---|---|---|
| `src/inference/raster.py` | 444 | Stage 7. Windowed inference over 20.76 M pixels in horizontal strips: read, encode, predict, write, discard. **No feature stack is ever materialised** — a 16-band float32 stack is ~1.25 GB and is exactly what blocked the previous attempt. Handles per-file nodata; a pixel is nodata if *any* contributing factor is. Supports clearly-suffixed partial runs when a raster is unavailable. |
| `src/decision/layers.py` | 423 | Stage 8. Rainfall intensity–duration component (`I = α·D^−β`, labelled uncalibrated everywhere it appears), exposure from roads + built-up LULC, the NISAR post-hoc multiplier slot, the priority surface, and the four-quadrant planner table. |
| `src/visualisation/maps.py` | 388 | Maps and charts. Single-hue sequential ramps for magnitude (never a rainbow), nodata rendered transparent (never a colour that could read as "low"), CVD-validated categorical palette with direct labels. |

### Tests

| File | Lines | What it does |
|---|---|---|
| `tests/test_config.py` | 295 | 20 tests. Config merge precedence, path resolution, output dirs, error messages. Critically: **`test_no_absolute_paths_in_source`** fails the build if a hardcoded path appears under `src/` — the regression test for the `D:\MMDIS\` failure — and **`test_themes_partition_the_factors_without_overlap`**, which protects the architecture's central claim. |

---

## 3. What gives more accuracy — ranked by measured effect

Every figure below is the measured change in the metric named.

| # | Change | Effect | Metric |
|---|---|---|---|
| 1 | **Setting the threshold instead of using 0.5** | **recall 0.338 → 0.650** | +0.312 recall |
| 2 | **Using all 14 factors vs the best single theme** | 0.5493 → 0.6564 | +0.107 AUC-PR |
| 3 | **Mixed learners per theme** *(not adopted — see §4)* | 0.6456 → 0.6557 | +0.0101 AUC-PR |
| 4 | Weighting the thematic blend by skill | 0.5697 → 0.6050 | +0.0353 AUC-PR |
| 5 | Blending the benchmark into the thematic mean | 0.6050 → 0.6456 | +0.0406 AUC-PR |
| 6 | Using CatBoost over XGBoost (full model) | 0.6468 → 0.6564 | +0.0096 AUC-PR |
| 7 | Using CatBoost over logistic (full model) | 0.5625 → 0.6564 | +0.0939 AUC-PR |
| 8 | **Not** using SMOTE | 0.6096 → 0.6564 | +0.0468 AUC-PR |
| 9 | **Not** using class weights | 0.6450 → 0.6564 | +0.0114 AUC-PR |

### The single biggest lever is the threshold, not the model

Everything in the modelling stack moves AUC-PR by hundredths. Choosing the
operating threshold properly moves **recall by 0.31** — roughly thirty times the
effect of any architectural decision. At the 0.5 default the model finds 141 of
417 known landslide locations; at 0.1374 it finds 271. For a system whose stated
cost asymmetry is that a missed zone costs far more than a false alarm, that is
the decision that matters.

### What *reduces* apparent accuracy but is correct

| Change | Effect | Why it is right |
|---|---|---|
| **Spatial block CV instead of random** | ROC-AUC 0.9132 → 0.8694 | The 0.9132 is measuring memorisation of neighbouring pixels. The lower number is the real one. |
| Dropping 4 impossible-elevation rows | negligible | They were DEM artefacts over water. |
| Calibrating before measuring disagreement | disagreement 0.0629 → 0.0522 | 17% of the "disagreement" was calibration mismatch. Removing it makes the confidence layer mean something. |

### Ranked feature importance (SHAP, full model)

| Feature | Share |
|---|---|
| `dist_road` | **27.5%** |
| `slope` | 11.4% |
| `rainfall` | 9.4% |
| `dist_fault` | 8.7% |
| `elevation` | 7.9% |
| `TWI` / `NDVI` / `lithology` | 4.7% / 4.6% / 4.6% |

**Read §6 before quoting `dist_road`.**

---

## 4. The final system

### Architecture

```
                    14 conditioning factors -> 16 encoded features
                                    |
        +---------------+-----------+-----------+---------------+
        |               |           |           |               |
   topographic     hydrological  geological  anthropogenic    full
     (7 feat)        (4 feat)     (2 feat)     (3 feat)     (16 feat)
     CatBoost        CatBoost     CatBoost     CatBoost     CatBoost
        |               |           |           |               |
        +-------- isotonic calibration (per model) -------------+
                        |                                       |
              mean(4 thematic)                          benchmark
                        |                                       |
                        +------- 0.5 x (full + mean) -----------+
                        |                                       |
                std(4 thematic)                          SUSCEPTIBILITY
                        |                                    raster
                  CONFIDENCE
                    raster
                        |
                        v
   susceptibility x confidence x exposure x nisar_multiplier = PRIORITY
```

### Settled parameters

| Parameter | Value | Chosen by |
|---|---|---|
| Learner | CatBoost, all 5 slots | Measured — beat XGBoost and logistic on the full model and 2 of 4 themes |
| CV | 5-fold spatial block, **10 km** | Block-size sweep; score stabilises 10–20 km |
| Imbalance | **None** | Measured — beat class weights and SMOTE on AUC-PR |
| Combination | `0.5 × (full + mean(thematic))` | Ablation — best ROC-AUC, −0.0024 AUC-PR vs best, and yields confidence |
| Calibration | Isotonic on spatially-blocked OOF | Required before disagreement is meaningful |
| Threshold | **0.1374** | Cost-minimising at 10:1; F2-optimal and Youden agree |
| Confidence | `1 − std(calibrated thematic) / p99` | — |

### The decision product

| Quadrant | Meaning | Area (provisional) |
|---|---|---|
| **Act now** | High susceptibility, themes agree | 10.8 km² |
| **Survey first** | High susceptibility, themes disagree | 758.1 km² |
| **Monitor** | Low susceptibility, themes disagree | 3,518.3 km² |
| **Deprioritise** | Low susceptibility, themes agree | 5,854.7 km² |

**Top 1% priority: 102.5 km²** out of 10,142 km².

### Why the uniform learner was kept over the mixed one

The mixed-learner ensemble scored **0.6557** — the best AUC-PR anywhere in this
study, beating even the full model alone. It was not adopted, and the reasoning
is on the record rather than hidden:

- The gain is **+0.0101 AUC-PR**, which is 8.6% of one fold standard deviation
  (±0.118). It is real but small relative to the noise.
- The cost is **cross-theme correlation rising 0.181 → 0.274, a 51% increase**.
  Disagreement then partly reflects "a boosted tree and a linear model behave
  differently" rather than "the physical evidence conflicts."
- The confidence raster is the thing the full-feature model structurally cannot
  produce. Degrading it by half to gain a tenth of a fold's noise is the wrong
  trade **for this project's stated objective**.

Both numbers are published. A reader who weights accuracy above decision quality
has what they need to choose otherwise, and the mixed configuration is a
one-line change.

---

## 5. Verification performed

| Check | Result |
|---|---|
| Unit tests | **20/20 pass** |
| No absolute paths under `src/` | Enforced by test; verified it still catches `D:\MMDIS\` |
| Themes partition the 14 factors exactly once | Enforced by test — passes |
| Raster grid alignment | 12/12 available rasters share one grid, matching `config.yaml` |
| District footprint | 10,929,202 px = 10,142 km² vs official 10,291 km² (1.4% — coastline clipping) |
| Inference pixel count | Exactly 10,929,202 — matches the audit independently |
| Label leakage | Proven: `taluk` positive rate 1.000 present / 0.000 absent. Blocked. |
| Rainfall raster ambiguity | Resolution code written; **pending** the missing raster |
| Determinism | Seed 42; stages 1–6 thread-pinned; every bundle records seed, strategy, CV scheme, library versions |
| Dependencies | Pinned exactly, resolved and verified as one set |
| Models loadable without their scaler | **Impossible by construction** — `ModelBundle` is the only persistence unit |

---

## 6. Honest weaknesses

Listed in order of how much they should worry you.

1. **`dist_road` dominance is partly circular.** It is 27.5% of model importance
   and the 3-feature anthropogenic theme nearly matches the 16-feature model.
   The district map visibly tracks the road network. This is partly real
   road-cut destabilisation and partly **inventory accessibility bias** — GSI
   field records under-report landslides in roadless forest. The two cannot be
   separated with this data. **The map must not be read as evidence that
   roadless terrain is safe.**

2. **The ≥101 m negative-sampling buffer is learnable.** The model can partly
   infer "distance from a known positive", which does not transfer to terrain
   where positives are unknown. Not fixable without resampling the negatives.

3. **417 positives for a 10,142 km² district** is a small positive class, and
   fold-to-fold AUC-PR spread is ±0.118. Differences below ~0.02 between
   configurations are not reliably differences.

4. **`dist_fault` is not fault data.** It is a DEM-derived structural-lineament
   proxy, so the geological and topographic themes share an upstream source.
   Measured consequence was small (correlation 0.116), but the label is
   misleading if quoted without the caveat.

5. **The rainfall threshold is uncalibrated.** Published regional parameters. A
   local calibration needs a rainfall time series that does not exist in the
   delivered assets.

6. **Exposure is coarse.** Built-up *land cover* at 30 m, no population
   weighting, no critical-infrastructure layer. The 0.6/0.4 weighting is assumed.

7. **Temporal mismatch.** NDVI, LULC and the 5-year rainfall mean are single
   snapshots; the inventory spans years. A slope forested when it failed may be
   mapped as bare today.

8. **30.46 m resolution.** Many Western Ghats failures, especially road-cuts,
   are narrower than one cell. The output identifies susceptible *zones*, not
   individual slopes.

9. **No independent temporal holdout.** All performance is cross-validated, not
   prospectively verified against post-inventory events.

10. **The lithology alarm was not confirmed.** The audit found a 27.4% raw
    landslide rate in the coverage gap; SHAP put lithology at only 4.6% of
    importance. The raw rate controlled for nothing. Recorded as a corrected
    finding rather than quietly dropped.

---

## 7. Outstanding

**Two of 14 rasters (`lithology_uk.tif`, `rainfall_uk_5yr_avg_aligned.tif`) and
the roads shapefile were not transferable during this session.** Sections 9–10
of `results.md`, and the rasters in `outputs/rasters/`, are therefore a
provisional 2-of-4-theme run, named with a `_partial_2of4themes` suffix so they
can never be mistaken for the deliverable.

Nothing about the modelling results depends on this — those come from the
training CSV, which is complete. To produce the full district map:

```bash
python run_pipeline.py check      # confirms all 14 rasters resolve
python run_pipeline.py stage 7 --to 8
python run_pipeline.py figures
```

No code change is required. The partial outputs can then be deleted.

---

## 8. Reproducing everything

```bash
pip install -r requirements.txt
cp config/config.local.yaml.example config/config.local.yaml   # set data_root
python run_pipeline.py check
python run_pipeline.py all
python run_pipeline.py figures
python -m pytest tests/ -q
```

Runtime: roughly 20 minutes end to end, dominated by the Stage 3 block-size
sweep and the Stage 4 learner comparison.

| Document | Contents |
|---|---|
| `README.md` | Problem, architecture, install, run |
| `docs/methodology.md` | Every decision, its justification, and the rejected alternatives |
| `docs/results.md` | Every measured number |
| `docs/data_dictionary.md` | Every column and layer: source, units, CRS, derivation, limitations |
| `docs/FINAL_AUDIT.md` | This document |
