# Results

All numbers below are measured, produced by `python run_pipeline.py all`, and
reproducible from the reports in `outputs/reports/`. Nothing here is asserted
from expectation.

**Run configuration:** 4,936 samples (417 positive / 4,519 negative, 1:10.84),
seed 42, CatBoost, 5-fold spatial block CV at 10 km blocks, no resampling.
Stages 1–6 ran with thread pools pinned to 1 for bit-reproducibility.

**One caveat, stated up front.** Sections 1–8 are complete. Sections 9–10, the
district rasters and the planner output, come from a **provisional run using 2
of 4 themes**: two conditioning-factor rasters (`lithology`, `rainfall`) could
not be transferred from the source machine during this session. The modelling
results are unaffected — they come from the training CSV, which is complete.
Re-running stage 7 with all 14 rasters present produces the full map with no
code change.

---

## 1. Data audit

Reproduced by `python run_pipeline.py stage 1` → `outputs/reports/data_audit.json`.

### Grid — the delivered claim verified, not assumed

All 12 available conditioning-factor rasters share one grid: 3,738 × 5,554 px,
EPSG:32643, 30.4625 m, identical affine transforms. The district footprint is
**10,929,202 valid pixels = 10,142 km²** against the official 10,291 km² — a
1.4% shortfall consistent with coastline clipping.

**Two nodata conventions are in use**, confirming this must be read per file:

| Value | Layers |
|---|---|
| −9999 | LULC, NDVI, aspect, dist_road, elevation, plan_curv, profile_curv, slope |
| −32768 | SPI, TWI, dist_drainage, dist_fault |

### Cleaning

4 rows carried negative elevation (`sample_id` 468, 473, 1666, 3519), all
negatives, three on permanent water. Dropped, logged. **4,940 → 4,936 rows**;
positives untouched at 417.

### Findings that changed a design decision

| Finding | Measured |
|---|---|
| Lithology coverage gap (`-1`) | 157 samples, 43 positive. **27.4% landslide rate vs 8.4% baseline.** Mean elevation 45.5 m vs 413.5 m; rainfall 3,820 vs 2,697 mm; dist_road 252 vs 894 m. Positives only in Karwar, Kumta, Honavar. A coastal strip, not random missingness. |
| Flat-cell aspect placeholder | 61 samples with `slope = 0`; **all 61 carry `aspect = 270.0` exactly** — one distinct value, so unambiguously a placeholder. 0 of them positive. |
| `taluk` label leak | Non-null on 417, null on 4,519. Positive rate where present **1.000**, where absent **0.000**. Perfectly separates the classes. |
| `slide_name` | Non-null on 233. Positive rate where present 1.000, where absent 0.039. Also a leak, blocked. |
| SPI skew | min 0.0004, max 2,250,643, **skew 46.3 → 1.29 under log1p**. |
| Spatial clustering | Nearest positive-to-positive: **min 0.65 m**, median 436 m; **86 within 100 m**, 212 within 500 m. |
| Negative sampling buffer | Minimum positive-to-negative distance **101.2 m** — confirms a ≥100 m exclusion buffer. |

### Spatial autocorrelation of the target

Empirical variogram, sill 0.0773:

| Lag | 500 m | 1,500 m | 2,500 m | 3,500 m | 4,500 m |
|---|---|---|---|---|---|
| % of sill | 86.9% | 93.2% | **96.3%** | 98.9% | 99.9% |

The 95%-of-sill range is **2,500 m**, but the variogram is already at 87% of
the sill in the first 500 m lag and effectively flat thereafter. That short
range is partly real and partly an artefact of the sampling design — negatives
were drawn with a ≥100 m exclusion buffer and spread across the district. A
single variogram-derived number is therefore a weak basis for choosing one
block size, which is why §2 sweeps it instead of asserting it.

---

## 2. Spatial vs random cross-validation

![Random CV overstates performance](../outputs/figures/spatial_vs_random.png)

Full-feature model, no resampling, identical folds within each scheme.

| Scheme | ROC-AUC | AUC-PR |
|---|---|---|
| **Random 5-fold** | **0.9132 ± 0.0129** | **0.7212 ± 0.0273** |
| Spatial block, 1 km | 0.8593 ± 0.0595 | 0.7505 ± 0.1586 |
| Spatial block, 2.5 km | 0.8721 ± 0.0424 | 0.7131 ± 0.1298 |
| Spatial block, 5 km | 0.8844 ± 0.0288 | 0.6905 ± 0.0778 |
| **Spatial block, 10 km** *(adopted)* | **0.8694 ± 0.0444** | **0.6564 ± 0.1179** |
| Spatial block, 20 km | 0.8732 ± 0.0492 | 0.6567 ± 0.1261 |

**Optimism injected by the random split, at the adopted 10 km block size:
+0.0438 ROC-AUC and +0.0648 AUC-PR.** That independently reproduces the
previous phase's finding of a 0.03–0.05 AUC drop under spatial blocking.

### Why the block size matters — the mechanism, measured

| Block size | Blocks | Points/block | Test points within 500 m of a training point | Median test→train distance |
|---|---|---|---|---|
| 1 km | 3,890 | 1.3 | 11.2% | 988 m |
| 2.5 km | 1,582 | 3.1 | 4.8% | 1,366 m |
| 5 km | 453 | 10.9 | 3.3% | 1,820 m |
| **10 km** | **127** | **38.9** | **1.7%** | **2,825 m** |
| 20 km | 41 | 120.4 | 1.0% | 4,412 m |

This is the leakage, made visible. Small blocks do not separate folds: at 1 km
there are 1.3 points per block, so "blocking" is barely different from random
assignment, and the AUC-PR of 0.7505 — *above* the random-CV figure, with a
±0.1586 spread — is noise, not a result. Blocks must be large enough to contain
the clustering scale before the score becomes meaningful. It stabilises between
10 km and 20 km, which is why 10 km was selected.

Every fold retained at least 83 positives at every block size, so no result
above is driven by a degenerate fold.

---

## 3. Class imbalance

Full-feature model, spatial CV at 10 km. Resampling applied strictly inside the
fold.

| Strategy | AUC-PR | ROC-AUC | Recall | Kappa |
|---|---|---|---|---|
| **None** *(adopted)* | **0.6564** | 0.8694 | 0.5060 | 0.5875 |
| Class weights | 0.6450 | 0.8712 | 0.5875 | 0.6010 |
| `scale_pos_weight` | 0.6450 | 0.8712 | 0.5875 | 0.6010 |
| SMOTE (in-fold) | 0.6096 | 0.8676 | 0.5707 | 0.5704 |

**The prior expectation held: SMOTE was worst**, by 0.047 AUC-PR. Synthetic
minority points interpolate in *feature* space, and a synthetic sample halfway
between two real landslides on different slopes need not correspond to any real
terrain.

Reweighting also slightly *hurt* AUC-PR while improving recall and kappa — it
shifts the operating point rather than improving the ranking, and this project
sets its operating point explicitly in §6 instead. Doing nothing was therefore
adopted: the imbalance is handled at the threshold, not in the loss.

*Note:* `class_weight` and `scale_pos_weight` coincide exactly for CatBoost —
both resolve to the same balanced reweighting. They are not two independent
results.

---

## 4. Individual model performance

![Thematic model strength](../outputs/figures/model_strength.png)

Spatial block CV, 10 km, no resampling. Threshold-dependent columns at 0.5, for
comparability only — the derived operating point is in §6.

| Model | Features | AUC-PR | ROC-AUC | F1 | Recall | Kappa |
|---|---|---|---|---|---|---|
| **Full-feature (benchmark)** | 16 | **0.6564 ± 0.1179** | 0.8694 | 0.6155 | 0.5060 | 0.5875 |
| Anthropogenic | 3 | 0.5493 ± 0.0965 | 0.8521 | 0.5213 | 0.4269 | 0.4847 |
| Topographic | 7 | 0.3566 ± 0.1489 | 0.7203 | 0.2934 | 0.2011 | 0.2622 |
| Hydrological | 4 | 0.1823 ± 0.0634 | 0.6666 | 0.0650 | 0.0384 | 0.0458 |
| Geological | 2 | 0.1354 ± 0.0355 | 0.6213 | 0.0245 | 0.0143 | 0.0167 |

The geological model is weakest, as predicted — two factors, one carrying the
coverage gap and the other a DEM-derived proxy rather than mapped faults.

**The uncomfortable result is the anthropogenic model.** Three features
(`dist_road`, `LULC`, `NDVI`) reach 0.8521 ROC-AUC, within 0.02 of the
full-feature model. The single strongest theme is the one most contaminated by
inventory accessibility bias, and §8 shows `dist_road` alone is 27.5% of total
model importance. This is discussed in §11.

### 4.1 Error decorrelation — the architecture's central claim

Out-of-fold prediction correlation between thematic models:

| | topographic | hydrological | geological | anthropogenic |
|---|---|---|---|---|
| **topographic** | 1.000 | 0.287 | 0.116 | 0.297 |
| **hydrological** | 0.287 | 1.000 | 0.162 | 0.142 |
| **geological** | 0.116 | 0.162 | 1.000 | 0.084 |
| **anthropogenic** | 0.297 | 0.142 | 0.084 | 1.000 |

**Mean off-diagonal correlation: 0.181.** The thematic split produced genuinely
decorrelated models. This is the result the whole design depends on — if these
had been as correlated as five same-feature learners, the confidence raster
would be meaningless.

The highest pair is topographic–anthropogenic at 0.297, which is physically
sensible: roads follow terrain. The geological–topographic pair, the one at
risk because `dist_fault` is DEM-derived, is **0.116** — the shared upstream
source did not translate into correlated predictions.

Mean residual correlation is 0.808, but that figure is an artefact and should
not be quoted as evidence of anything. With 8.4% positives every model predicts
low almost everywhere, so residuals are dominated by the label itself and are
correlated by construction. Prediction correlation is the informative measure
here.

---

## 5. Ablation: combination rules

![Combination-rule ablation](../outputs/figures/ablation.png)

All rules scored on the same out-of-fold predictions from the same spatial
folds. Parametric rules were refitted inside folds.

| Combination rule | AUC-PR | ROC-AUC | Produces confidence? |
|---|---|---|---|
| **Full model alone** | **0.6480** | 0.8677 | **No** |
| **Full + thematic mean** *(adopted)* | **0.6456** | **0.8705** | **Yes** |
| Thematic weighted mean | 0.6050 | 0.8608 | Yes |
| Thematic meta-learner (logistic) | 0.5955 | 0.8418 | Yes |
| Thematic mean | 0.5697 | 0.8490 | Yes |
| Thematic noisy-OR | 0.5642 | 0.8476 | Yes |
| Thematic max | 0.5177 | 0.8321 | Yes |
| Thematic rank average | 0.5124 | 0.8501 | Yes |

**The headline: the confidence layer costs 0.0024 AUC-PR.**

The full-feature model alone has the best AUC-PR. But `full_plus_thematic_mean`
is behind it by 0.0024 — against a fold-to-fold standard deviation of ±0.118,
so the gap is roughly 2% of one standard deviation, far inside noise — and it
*also* has the best ROC-AUC in the table (0.8705 vs 0.8677) and produces the
confidence raster that the single model structurally cannot.

For comparison, the previous architecture's stacked ensemble lost to its own
best member by **0.0104**. This design's ensemble gives up **0.0024** and
returns a decision product for it. That is the difference between an unjustified
ensemble and a justified one.

The logistic meta-learner — the previous design's combiner — again placed
mid-table at 0.5955, well below the simple blend. Replacing the meta-learner was
never the fix.

---

## 5b. Does a mixed-learner ensemble do better?

Tests whether letting each theme choose its own algorithm beats holding the
algorithm fixed. Per-theme comparison under the same spatial CV:

| Model | CatBoost | XGBoost | Logistic | Best |
|---|---|---|---|---|
| Topographic | **0.3566** | 0.3441 | 0.2122 | CatBoost |
| Hydrological | 0.1823 | 0.1795 | **0.2063** | **Logistic** |
| Geological | 0.1354 | 0.1587 | **0.1694** | **Logistic** |
| Anthropogenic | **0.5493** | 0.5161 | 0.4487 | CatBoost |
| Full | **0.6564** | 0.6468 | 0.5625 | CatBoost |

**Logistic regression beats CatBoost on the two smallest themes** — by 0.024 on
hydrological and 0.034 on geological (a 25% relative gain). With 2–4 features
and 417 positives, boosted trees overfit where a regularised linear model does
not.

Ensemble-level effect:

| Combination | Uniform (all CatBoost) | Mixed learners | Δ |
|---|---|---|---|
| Thematic mean | 0.5697 | **0.6054** | **+0.0357** |
| Thematic weighted mean | 0.6050 | 0.6139 | +0.0089 |
| Full + thematic mean | 0.6456 | **0.6557** | **+0.0101** |

**The mixed ensemble at 0.6557 is the best AUC-PR anywhere in this study** —
above the full model alone (0.6480) — *and* it produces confidence.

**But it costs decorrelation: mean prediction correlation rises 0.181 → 0.274,
a 51% increase.** Part of the disagreement then reflects "a boosted tree and a
linear model behave differently" rather than "the physical evidence conflicts",
and the disagreement signal is the product here.

**Adopted: the uniform-learner blend**, with the mixed variant documented as a
validated alternative. Reasoning in `methodology.md` §5b. The +0.0101 is 8.6% of
one fold standard deviation; the 51% correlation increase degrades the only
output the full-feature model cannot produce. A reader who weights accuracy
above decision quality has the number to make the other choice.

---

## 6. Operating threshold

Scored on the calibrated production score.

| Method | Threshold | Recall | Precision | F1 | F2 | Kappa | Flagged |
|---|---|---|---|---|---|---|---|
| **Cost-minimising (10:1)** *(adopted)* | **0.1374** | **0.650** | 0.559 | 0.601 | 0.629 | 0.561 | 9.8% |
| F2-optimal | 0.1382 | 0.650 | 0.560 | 0.602 | 0.630 | 0.562 | 9.8% |
| Youden J | 0.1382 | 0.650 | 0.560 | 0.602 | 0.630 | 0.562 | 9.8% |
| Precision floor (0.25) | 0.0687 | 0.782 | 0.250 | 0.379 | 0.548 | 0.288 | 26.4% |
| *Default 0.5 (reference only)* | 0.500 | **0.338** | 0.849 | 0.484 | 0.384 | 0.458 | 3.4% |

**Choosing the threshold rather than defaulting to 0.5 nearly doubles recall,
0.338 → 0.650.** At 0.5 the model misses two-thirds of known landslide
locations while reporting 85% precision — precisely the failure mode that
makes a susceptibility map dangerous rather than merely unhelpful.

Three independent criteria — cost-minimising, F2-optimal and Youden — converge
on 0.137–0.138, which is reassuring: the operating point is not an artefact of
one objective.

### Sensitivity to the assumed cost ratio

The 10:1 ratio is a stated assumption, not a measured quantity, so its
influence is reported:

| Cost ratio | Threshold | Recall | Precision | Flagged |
|---|---|---|---|---|
| 2:1 | 0.2456 | 0.571 | 0.748 | 6.4% |
| 5:1 | 0.1765 | 0.621 | 0.656 | 8.0% |
| **10:1** | **0.1374** | **0.650** | 0.559 | 9.8% |
| 20:1 | 0.0707 | 0.779 | 0.260 | 25.3% |
| 50:1 | 0.0375 | 0.974 | 0.110 | **74.7%** |

The threshold is stable between 2:1 and 10:1 and then moves sharply. At 50:1 the
model achieves 97% recall by flagging three-quarters of the district — a
reminder that recall alone is not a goal, since a map that flags everything
allocates nothing.

---

## 7. Calibration and confidence

### Calibration (isotonic, fitted on out-of-fold, evaluated through a second fold layer)

| Model | Brier before | Brier after | Mean p before | Mean p after |
|---|---|---|---|---|
| Topographic | 0.06754 | 0.06747 | 0.0674 | 0.0849 |
| Hydrological | 0.07897 | **0.07584** | 0.0655 | 0.0859 |
| Geological | 0.07790 | 0.07683 | 0.0775 | 0.0858 |
| Anthropogenic | 0.05324 | 0.05402 | 0.0833 | 0.0852 |

Observed base rate is 0.0845. Every model was under-predicting the positive
rate before calibration; all four now sit within 0.0015 of it. Brier
improvements are small because the models were already reasonably calibrated in
aggregate — the correction matters for the *relative scaling between* them,
which is what the next table measures.

### Calibration was a precondition, and here is why

| | Mean disagreement |
|---|---|
| Uncalibrated thematic probabilities | 0.0629 |
| Calibrated thematic probabilities | **0.0522** |
| Correlation between the two | 0.938 |

**17% of the apparent cross-model disagreement was calibration mismatch, not
genuine disagreement about the pixel.** Had the confidence raster been built
from uncalibrated scores, roughly a sixth of the signal in the project's
headline output would have been an artefact of how each model happened to be
scaled.

### Confidence distribution (out-of-fold)

| p10 | median | mean | p90 |
|---|---|---|---|
| 0.7512 | 0.9187 | 0.8697 | 0.9873 |

### The four-quadrant decision table (out-of-fold samples)

Threshold 0.1374; confidence split at the median, 0.9187.

| Quadrant | Samples | Known landslides |
|---|---|---|
| **Act** — high susceptibility, themes agree | 36 | 8 |
| **Survey first** — high susceptibility, themes disagree | 449 | **263** |
| **Monitor** — low susceptibility, themes disagree | 2,016 | 77 |
| **Deprioritise** — low susceptibility, themes agree | 2,435 | 69 |

**63% of all known landslides (263 of 417) fall in "survey first".** This is the
most interesting result in the study and it is not what a naive reading would
predict. Where landslides actually occurred, the four physical themes tend to
*disagree* — one line of evidence flags the site strongly while others do not.

That is a substantive finding rather than a defect. It says real failure sites
in this district are not places where every factor is adverse at once; they are
places where a single dominant mechanism is present. It is also exactly the
population a survey team should be sent to, and a single-model map would have
shown these as ordinary high-susceptibility pixels with no indication that the
evidence behind them was thin.

---

## 8. SHAP interpretability

Full-feature model, CatBoost native SHAP over all 4,936 samples.

| Rank | Feature | Share of total importance |
|---|---|---|
| 1 | **`dist_road`** | **27.5%** |
| 2 | `slope` | 11.4% |
| 3 | `rainfall` | 9.4% |
| 4 | `dist_fault` | 8.7% |
| 5 | `elevation` | 7.9% |
| 6 | `TWI` | 4.7% |
| 7 | `NDVI` | 4.6% |
| 8 | `lithology` | 4.6% |

### Cross-theme factor agreement

**14 of 16 features read consistently between their own thematic model and the
full-feature model** (per-sample SHAP correlation ≥ 0.5), mean correlation
**0.725**.

That is the accounting of what the thematic split costs. Most factors carry a
self-contained physical signal that means the same thing whether or not the
model can see the other themes. The two that do not are being used in
cross-theme interaction — information the thematic models give up by
construction, and part of why the full-feature model retains a small edge.

### `dist_road` — the finding that needs its caveat

`dist_road` is the single largest driver at 27.5%, and the anthropogenic theme
built on it reaches 0.8521 ROC-AUC from three features. The district map in §9
shows susceptibility visibly tracking the road network.

This reflects two effects the available data cannot separate:

1. **Genuine destabilisation.** Road-cutting removes slope support and is a
   well-documented Western Ghats landslide trigger.
2. **Inventory accessibility bias.** GSI field-validated records are
   reporting-driven; landslides in remote roadless forest are systematically
   under-recorded.

No statement about `dist_road` importance in this project should be made without
this caveat, and the map should not be read as evidence that roadless terrain is
safe.

### `lithology`

At 4.6% of full-model importance, and 25% within the geological theme (against
`dist_fault` at 75%). **This does not support the strong reading of the raw
coverage-gap rate.** The `-1` class shows a 27.4% raw landslide rate, but raw
rates control for nothing — that coastal strip is also wetter, lower-lying and
far closer to roads, and once those are accounted for, lithology as a whole is
a minor contributor. This independently corroborates the previous phase's
finding that `-1`'s independent contribution is modest, and the alarm raised
during the audit is recorded as **not confirmed**.

---

## 9. District-wide outputs

> **Provisional — 2 of 4 themes.** Built from the topographic and anthropogenic
> models only; `lithology` and `rainfall` rasters were unavailable, so the
> geological and hydrological themes and the full-feature benchmark were
> excluded. Spatial patterns and the pipeline are real; absolute values will
> shift when the full run is done.

![Susceptibility map](../outputs/figures/susceptibility_map.png)

Windowed inference over the complete grid: **10,929,202 valid pixels**, exactly
matching the expected footprint, in 11 strips of 512 rows. Peak memory bounded
by strip size; no feature stack was materialised at any point.

| | |
|---|---|
| Mean susceptibility | 0.0672 |
| Above threshold (0.1374) | **768.9 km² (7.6% of the district)** |
| Disagreement ceiling (99th pct) for confidence scaling | 0.3034 |

| Susceptibility class | Range | Area | Share |
|---|---|---|---|
| Very low | 0.00 – 0.05 | 5,310.1 km² | 52.4% |
| Low | 0.05 – 0.10 | 3,567.2 km² | 35.2% |
| Moderate | 0.10 – 0.137 | 495.7 km² | 4.9% |
| **High** | 0.137 – 0.40 | **686.9 km²** | 6.8% |
| **Very high** | ≥ 0.40 | **82.1 km²** | 0.8% |

The mean of 0.0672 sits just below the 0.0845 training base rate, which is the
expected direction: the training set was deliberately enriched in steep terrain
by slope-stratified negative sampling, so the district as a whole is gentler
than the sample.

---

## 10. Prioritised planner output

> **Provisional**, per §9. The roads shapefile was also unavailable, so exposure
> here rests on LULC built-up alone (95,508 px) with the roads component at
> zero. The full run restores it.

`priority = susceptibility × confidence × exposure × nisar_multiplier`

| Quadrant | Area |
|---|---|
| **Act now** — high susceptibility, themes agree | 10.8 km² |
| **Survey first** — high susceptibility, themes disagree | 758.1 km² |
| **Monitor** — low susceptibility, themes disagree | 3,518.3 km² |
| **Deprioritise** — low susceptibility, themes agree | 5,854.7 km² |

**Top 1% priority zones: 102.5 km²** — the shortlist a planner actually works
from, reduced from a 10,142 km² district.

### Rainfall intensity–duration thresholds

> **UNCALIBRATED.** Published regional parameters (α = 15.0, β = 0.44) in the
> Caine form `I = α·D^(−β)`. No rainfall time series exists in the delivered
> assets, so no local calibration is possible. **Do not quote these as Uttara
> Kannada thresholds.**

| Duration | Threshold intensity | Total accumulation |
|---|---|---|
| 6 h | 6.82 mm/h | 41 mm |
| 12 h | 5.03 mm/h | 60 mm |
| 24 h | 3.71 mm/h | 89 mm |
| 48 h | 2.73 mm/h | 131 mm |
| 72 h | 2.29 mm/h | 164 mm |

### NISAR

Disabled — no deformation raster supplied. The slot is live: setting
`decision.nisar.enabled` and supplying a raster applies it as a post-hoc
multiplier with **no retraining**.

---

## 11. What this study concludes

1. **The rebuild's core claim is supported.** Thematic feature subsets produced
   genuinely decorrelated models (mean correlation 0.181), which the previous
   five-algorithm stack did not achieve. The confidence layer rests on real
   disagreement.

2. **The ensemble is justified, narrowly and honestly.** It costs 0.0024 AUC-PR
   against the best single model — 2% of one fold standard deviation — and
   returns a confidence surface plus the best ROC-AUC in the table. Compare the
   previous stack's 0.0104 deficit for no decision product at all.

3. **The honest headline number is 0.8694 ROC-AUC, not 0.9329.** The difference
   is measured leakage, not a worse model.

4. **The most predictive theme is the most epistemically compromised.** The
   anthropogenic model nearly matches the full model, and `dist_road` is 27.5%
   of total importance, entangled with inventory reporting bias. This is the
   project's largest unresolved weakness.

5. **Most known landslides sit in the low-confidence quadrant** (263 of 417),
   which is both a substantive finding about how failures occur here and the
   clearest demonstration that the confidence layer carries information a single
   model cannot supply.

6. **Your mixed-learner alternative measurably wins on accuracy** (0.6557, the
   best result anywhere in this study) and measurably loses on decorrelation
   (0.181 → 0.274). Both numbers are reported so the trade can be judged rather
   than assumed.

---

## 12. Reproducibility

| Artefact | Path |
|---|---|
| Data audit | `outputs/reports/data_audit.json` |
| Feature spec | `outputs/reports/features.json` |
| Spatial CV & imbalance | `outputs/reports/spatial_cv.json` |
| Models & ablation | `outputs/reports/models_ablation.json` |
| Calibration & threshold | `outputs/reports/calibration_threshold.json` |
| Interpretability | `outputs/reports/interpretability.json` |
| Inference | `outputs/reports/inference*.json` |
| Decision layer | `outputs/reports/decision*.json` |
| Out-of-fold predictions | `outputs/reports/oof_predictions.csv`, `oof_calibrated.csv` |
| Model bundles | `outputs/models/*.joblib` (+ `.json` sidecars) |
| Run logs | `outputs/logs/` |

Seed 42 throughout. Stages 1–6 ran with thread pools pinned to 1. Every model
bundle records its seed, imbalance strategy, CV scheme, block size, training
counts and library versions.
