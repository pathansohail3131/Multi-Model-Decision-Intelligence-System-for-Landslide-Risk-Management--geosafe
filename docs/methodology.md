# Methodology

Every methodological decision in this project, with its justification and the
alternatives that were rejected. Written as the project is built, not
reconstructed afterwards.

**Status: all 8 stages executed.** Every decision below now carries its
measured outcome. Where a prediction made in advance was wrong, it is marked as
corrected rather than quietly revised. Full numbers live in
[`results.md`](results.md).

---

## 0. Framing

### 0.1 The decision problem is not the classification problem

The classification task is close to solved. The previous version's CatBoost
model reached 0.9329 ROC-AUC, and marginal gains beyond that are largely noise
against the uncertainty already present in the inventory.

The unsolved problem is that a probability surface is not a decision surface. A
planner with fixed survey budget cannot act on "0.78". The design goal is
therefore **decision-relevant output**: susceptibility *plus* confidence *plus*
exposure, with physical attribution.

This reframing has a concrete consequence for model selection. Under a pure
accuracy objective, if the full-feature model outperforms the thematic
ensemble, the ensemble should be discarded. Under a decision objective, the
ensemble may still be preferred, because the full-feature model structurally
cannot produce a disagreement signal — there is nothing for it to disagree
with. The ablation in [`results.md`](results.md) reports the size of that
tradeoff so it is a judgement made in the open, not a preference smuggled in.

### 0.2 Why the previous architecture failed

Five algorithms (RF, XGBoost, CatBoost, SVM, ANN) on identical features,
combined by a logistic-regression meta-learner. Result: 0.9225 stacked against
0.9329 for CatBoost alone.

The diagnosis is structural, not a tuning failure:

- Three of the five (RF, XGBoost, CatBoost) are tree ensembles trained on the
  same columns. They make correlated errors. Averaging correlated predictors
  reduces variance only to the extent the errors are independent; when they are
  not, it adds estimation noise without adding information.
- The two structurally different learners (SVM, ANN) were substantially weaker,
  so the meta-learner could not usefully upweight them.
- Stacking requires members that are right in *different places*. These members
  were right in the same places.

**Rejected fix: a better meta-learner.** Replacing logistic regression with a
gradient-boosted meta-learner, or tuning the stack, addresses none of the
above. The members' errors would still be correlated. The problem is the
ensemble's composition, not its combiner.

**Adopted fix:** force diversity through the *feature space* instead of the
algorithm space. See §3.

---

## 1. Engineering decisions

### 1.1 Configuration and paths

**Decision.** A single `data_root` is the only machine-specific value.
Precedence: `LANDSLIDE_DATA_ROOT` env var → `config/config.local.yaml`
(gitignored) → `config/config.yaml` (tracked). No module under `src/` contains
an absolute path, enforced by
`tests/test_config.py::test_no_absolute_paths_in_source`.

**Why.** The previous version hardcoded `D:\MMDIS\` in notebook cells. It ran
on one machine, and when that machine changed every path broke at once with no
single place to fix them.

**Rejected: relative paths only, data inside the repo.** Simplest, but the
assets are ~1.2 GB including a >1 GB GLiM geodatabase. Committing them, or
requiring them to be copied into the tree, is worse than one configurable root.

**Rejected: a `.env` file.** Equivalent in effect, but adds a dependency and
splits configuration across two formats for one value.

### 1.2 Pinned dependencies

**Decision.** Exact pins (`==`) for every dependency, resolved and verified as
a set.

**Why.** The previous version had no requirements file, and its pickled models
became unloadable after scikit-learn and CatBoost moved on — pickled estimators
embed class layouts that do not survive a major version bump. A susceptibility
model that cannot be loaded cannot be defended.

**Rejected: compatible-release pins (`~=`).** Friendlier to future upgrades,
but permits exactly the silent minor-version drift that produced the original
failure.

**Consequence.** TensorFlow and Keras are dropped entirely, since the rebuilt
architecture uses no ANN. That removes a ~600 MB dependency and a substantial
source of nondeterminism.

### 1.3 Determinism

**Decision.** One seed propagates to Python, NumPy and every estimator, with
`random_state` passed explicitly rather than inherited from a global RNG.
Stages 1–6 additionally pin BLAS/OpenMP thread counts to 1; stages 7–8 do not.

**Why.** The headline ablation gap is on the order of 0.01 AUC. Parallel
floating-point reduction is not associative, so thread-count variation alone
can move a metric by a comparable amount. A result that cannot be reproduced to
more precision than the effect it claims is not a result.

The relaxation for stages 7–8 is deliberate: their output is a smooth
probability surface across 10.9 M pixels, where a last-bit difference is
invisible and the speed matters.

### 1.4 Models are serialised with their preprocessing

**Decision.** Every persisted model bundles its fitted transformer.

**Why.** A model that *can* be loaded without its scaler eventually *will* be,
and the resulting predictions look completely plausible while being wrong. This
is a particularly dangerous failure for raster output, because there is no
obvious visual signature — the map still looks like a map. The previous
version's SVM and ANN required separately applied preprocessors, and its
meta-learner required base predictions in one specific undocumented order
(RF, XGB, CB, SVM, ANN), which is the same class of latent hazard.

---

## 2. Data decisions

The audit behind these is reproduced by `python run_pipeline.py stage 1`.
Per-column detail is in [`data_dictionary.md`](data_dictionary.md).

### 2.1 Lithology coverage gap (`lithology = -1`)

**Finding.** 157 of 4,940 samples carry the sentinel `-1`, absent from the
five-class lookup in `lithology_classes.json`. They are **not** randomly
missing:

| | Samples with `lithology = -1` | All samples |
|---|---|---|
| Count | 157 | 4,940 |
| Landslide rate | 27.4% | 8.4% |
| Mean elevation | 45.5 m | 413.5 m |
| Mean rainfall | 3,820 mm | 2,697 mm |
| Mean distance to road | 252 m | 894 m |
| Taluks (positives) | Karwar, Kumta, Honavar only | all ten |

They form a low-lying coastal strip where GLiM polygon coverage is absent. The
sentinel is therefore spatially confounded with the outcome: a model given this
code learns "unmapped → landslide", which is really "coastal → landslide".

**Decision.** Retain as an explicit named category
(`unmapped_coastal_gap`), and disclose it.

**Why.** The confounding is real but so is the hazard. The coastal Karwar–Kumta
belt is where NH-66 landslides actually recur, and it contains 43 of 417
positives.

Crucially, the thematic architecture **contains** the damage. Only the
geological model sees `lithology`; the topographic, hydrological and
anthropogenic models are unaffected. Where the geological model is leaning on
a coverage artefact, it should *disagree* with the other three — which the
confidence raster will surface as low confidence rather than hide inside an
averaged probability. This is an argument for the thematic split, not merely a
caveat about it.

**Rejected: drop the 157 rows.** Yields the cleanest geological signal, but
deletes 10% of all positives and does so *systematically* — removing the entire
coastal hazard belt rather than a random subset. The resulting map would be
confidently wrong exactly where the district's road network is most exposed.

**Rejected: impute from the nearest mapped polygon.** Keeps every sample and
removes the artefact, but fabricates geology that GLiM never mapped, and writes
it into a document intended to be defended. The invented values would also be
systematically wrong in the same direction, since the nearest mapped polygon to
a coastal gap is by construction an inland unit.

**Limitation recorded.** The geological model's behaviour in the coastal strip
partly reflects data coverage rather than geology. Any SHAP attribution to
`lithology = unmapped_coastal_gap` must be read as a coverage indicator.

**Counter-evidence, recorded before Stage 6 so it cannot be quietly dropped.**
The previous phase's SHAP analysis found that `-1`'s *independent* contribution
to predicted risk was only modest once other factors were held constant, and
that the larger, clearer lithology signal ran the other way — classes 3
(siliciclastic sedimentary) and 4 (basic volcanic) were substantially
*protective*. The 27.4% figure above is a **raw rate that controls for nothing**;
the coastal strip is also wetter, steeper-adjacent and far closer to roads, and
those covariates may account for most of the apparent effect.

So the retain-and-disclose decision stands on its own merits — it avoids both
deleting the coastal hazard belt and fabricating geology — but the strength of
the confounding it guards against is an **open empirical question**, not an
established finding. Stage 6 answers it. If SHAP again shows a modest
independent contribution, this section is updated to say so rather than left
implying a problem that the models do not actually have.

**Measured outcome: the alarm is NOT confirmed, and this section is duly
corrected.** SHAP on the full-feature model puts `lithology` at **4.6% of total
importance** — eighth of sixteen features — and within the geological theme it
carries only 25% against `dist_fault`'s 75%. The 27.4% raw landslide rate in the
coverage gap is real as a raw rate, but raw rates control for nothing: that
coastal strip is also wetter, lower-lying and far closer to roads, and once
those are accounted for, lithology is a minor contributor overall.

The retain-and-disclose decision still stands, because it was never contingent
on the confounding being severe — it avoided both deleting 10% of the positives
and fabricating geology. But the strength of the concern was overstated during
the audit, and the record says so.

### 2.2 Aspect is circular, and flat cells have no aspect

**Finding.** Two separate issues in one column.

1. Aspect is measured in degrees on a circle. Encoded as a scalar, 359° and 1°
   are 358 units apart despite being adjacent, and a tree must spend splits
   reassembling a continuity that the encoding destroyed.
2. Every sample with `slope == 0` (61 rows) carries `aspect = 270.0` **exactly**
   — the terrain software's placeholder for undefined aspect on a flat cell,
   not a measured direction.

**Decision.** Encode as `aspect_sin` / `aspect_cos`, and add a binary `is_flat`
flag; for flat cells, zero both components.

**Why.** sin/cos places north adjacent to north. Zeroing the components on flat
cells places them at the circle's origin rather than at a spurious due-west
bearing, and `is_flat` lets the model represent flatness explicitly instead of
inferring it from a magic value.

This makes **16** model features from 14 conditioning factors, where the
original brief implies 15. The extra column is `is_flat`, and it exists because
without it the placeholder 270° is indistinguishable from a genuine westward
slope.

**Rejected: leave aspect as degrees.** Tree models can in principle carve a
circular variable into bins, but it costs depth, and it makes SHAP attribution
for aspect close to unreadable.

**Rejected: one-hot the eight compass sectors.** Discards within-sector
gradient and puts an arbitrary boundary at each sector edge.

**Rejected: drop flat cells.** Flat terrain is genuinely low-susceptibility and
is legitimate negative evidence; discarding it would bias the negative class
toward sloped ground and flatter the model's apparent discrimination.

### 2.3 `taluk` and `slide_name` are label leaks

**Finding.** Both columns are non-null on all 417 positives and null on all
4,523 negatives. Either one perfectly separates the classes.

**Decision.** Listed in `features.forbidden_columns` and blocked at the feature
layer rather than dropped from the dataframe, so they remain available for
reporting and stratification diagnostics. A test enforces the block.

**Why.** They are provenance metadata from the inventory, not measured
conditions. Blocking rather than deleting keeps the audit trail while making
accidental use impossible.

**Consequence.** `taluk` cannot be the spatial blocking unit either, because it
is unavailable for the negative class. Blocking uses coordinates directly
(§4.1).

### 2.4 `SPI` is log-transformed

**Finding.** Stream Power Index ranges from 0.0004 to 2,250,643 with skew 46.3.
`log1p` brings skew to 1.29.

**Decision.** Model on `log1p(SPI)`.

**Why.** SPI is a product of catchment area and slope gradient and is
multiplicative by construction, so its natural scale is logarithmic.

**Note on necessity.** Gradient-boosted trees are invariant to monotonic
transforms, so this does not change tree-model performance. It is retained
because it makes SHAP values and partial dependence readable on a physically
meaningful scale, and because it keeps the door open to any scale-sensitive
component without a second, inconsistent preprocessing path.

### 2.5 Nodata is read per file

**Finding.** The GDAL-derived layers use −9999; the WhiteboxTools-derived
hydrological layers (TWI, SPI, flow accumulation, distance-to-drainage,
distance-to-fault) use −32768.

**Decision.** Each raster's nodata is read from its own header. No global
nodata constant exists in the codebase.

**Why.** Assuming a single value would admit −32768 into the feature matrix as
a real measurement wherever the other convention is used. The failure is
silent: inference completes, the raster is written, and the map looks fine.

### 2.6 Rainfall layer ambiguity

**Finding.** `data/processed/` contains both `rainfall_uk_aligned.tif` and
`rainfall_uk_5yr_avg_aligned.tif`. The brief specifies the 5-year average.

**Decision.** Configured as the 5-year average; **Stage 1 verifies which layer
the training CSV was actually sampled from** by re-sampling both at the
training coordinates and comparing.

**Why.** If the CSV was built from one layer and inference runs on the other,
training and prediction sit on different distributions. This produces no error
and no visible artefact — only a quietly wrong map.

### 2.7 Spatial clustering and the negative-sampling buffer

**Finding.** 86 of 417 positives lie within 100 m of another positive
(minimum separation 0.65 m). Negatives were drawn with a ≥101 m buffer from any
positive; the observed minimum positive-to-negative distance is 101.2 m.

**Decision.** Spatial block CV as the evaluation of record (§4.1). The buffer
is recorded as a limitation, not corrected.

**Why.** The clustering is what makes random CV dishonest here. The buffer is a
defensible sampling choice — a "non-landslide" point 5 m from a landslide is
not credibly a negative — but it does mean the model can partly learn "distance
from the nearest positive", which will not generalise to a district where
positives are unknown. It cannot be fixed without resampling the negatives,
which is out of scope for a rebuild that treats the data as validated input.

**Positive note.** The slope stratification worked: positives average 13.3°
against 8.4° for negatives, overlapping rather than separable. The negative
class is honest — it is not padded with flat coastal plain that any model could
trivially reject.

### 2.8 The delivered CSV is the pre-cleanup file

**Finding.** The brief describes 4,940 samples, and the delivered
`training_dataset_final.csv` contains exactly that. But 4 of those rows carry
physically impossible negative elevation (−1 m to −3 m), and three of the four
sit on LULC class 80 (permanent water) near the estuaries — a confirmed DEM
interpolation artefact over water. The previous phase dropped these at load
time and modelled on 4,936 rows; the file itself was never rewritten.

**Decision.** Drop the 4 rows explicitly in Stage 1, logged by `sample_id`, for
a working dataset of **4,936 rows — 417 positive / 4,519 negative**.

**Why.** All four are negatives, so no positive evidence is lost, and a
negative elevation is not a plausible measurement. Doing it in the pipeline
with a logged reason — rather than silently at load, or by editing the source
CSV — keeps the delivered asset untouched and the removal auditable.

**Consequence for reporting.** Any figure quoted as "4,940 samples" or
"1:10.85" describes the file, not the model. The model is trained on 4,936 at
1:10.84. The difference is immaterial to performance and material to
reproducibility, so both numbers appear in `results.md`.

**Open question for Stage 1: LULC class 80 (permanent water).** 105 negative
samples and zero positives sit on permanent water. Water cannot fail, so these
are trivially separable negatives that inflate every metric by a small amount
while teaching the model nothing. Whether to exclude them is a genuine tradeoff
— at inference those pixels exist in the district and must receive *some*
value, and excluding them from training means predicting on a class the model
never saw. Deferred to Stage 1 with the decision and its evidence recorded
there.

---

## 3. Model architecture

### 3.1 Thematic feature subsets

**Decision.** One strong learner per theme — topographic, hydrological,
geological, anthropogenic/surface — plus a full-feature benchmark. The themes
partition the 14 factors exactly once each, enforced by
`tests/test_config.py::test_themes_partition_the_factors_without_overlap`.

**Why.** Diversity is induced by restricting what each model can see, which is
a structural guarantee rather than an assumption. Two models sharing no inputs
cannot make correlated errors *through a shared input*, which is exactly the
mechanism that defeated the previous stack.

The partition is also physically meaningful, which matters for the decision
product: "flagged by terrain but not by geology" names a specific kind of
uncertainty a survey team can be sent to resolve.

**Known consequence: the geological model will be weak.** It sees only two
factors, `lithology` and `dist_fault`, one of which has the coverage gap in
§2.1. This is expected and is not a reason to rebalance the themes — the themes
follow the physics, not the feature count. A weak-but-independent member is
still informative for disagreement, and the ablation (§5) measures whether it
earns its place in the aggregate.

**Rejected: equal-sized thematic subsets.** Would require splitting themes
across physical meaning, destroying the interpretability that is the point.

**Rejected: overlapping subsets so every model sees slope.** Slope is the
strongest single predictor and sharing it would lift every thematic model's
individual score — while reintroducing the correlated-error problem the design
exists to remove. Agreement between models that share the dominant feature
measures the feature, not the evidence.

**Caveat: the partition is clean in columns but not fully clean in provenance.**
`dist_fault` is not distance to mapped faults — no field-validated fault or
lineament map exists for this district at usable resolution, so the layer is a
**DEM-derived structural-lineament proxy**
([`data_dictionary.md`](data_dictionary.md) §3.3). It is therefore derived from
the same elevation surface as slope, curvature and aspect. The geological and
topographic models share no *column*, but they do share an *upstream source*,
which weakens the blindness the architecture depends on for exactly that pair.

This is disclosed rather than engineered around, because the alternative —
dropping `dist_fault` — would leave the geological theme with a single feature
carrying a coverage gap, which is worse. Stage 4 quantifies the damage by
measuring the topographic–geological prediction correlation directly
([`results.md`](results.md) §4.1) instead of assuming it is small.

### 3.2 Learner choice

**Decision: CatBoost, in all five slots. Confirmed empirically.**

Measured under spatial block CV (AUC-PR):

| Model | CatBoost | XGBoost | Logistic |
|---|---|---|---|
| Full-feature | **0.6564** | 0.6468 | 0.5625 |
| Topographic | **0.3566** | 0.3441 | 0.2122 |
| Anthropogenic | **0.5493** | 0.5161 | 0.4487 |
| Hydrological | 0.1823 | 0.1795 | **0.2063** |
| Geological | 0.1354 | 0.1587 | **0.1694** |

CatBoost wins the full model and the two larger themes. The prior favourite was
confirmed rather than assumed — and the two exceptions are discussed in §5b,
because they are not a footnote.

**Rejected: per-theme learner selection (as a default).** It measurably improves
the ensemble but degrades the disagreement signal. Full reasoning and numbers in
§5b.

### 3.3 Categorical handling

**Decision.** `lithology` and `LULC` are passed to the learner as categorical
features, never label-encoded into a model that would read the integer codes as
an ordering.

**Why.** The codes are nominal. LULC classes (10 tree cover, 20 shrubland, 30
grassland, 40 cropland, 50 built-up, 60 bare/sparse, 80 water, 90 wetland,
95 mangrove) have no ordering; treating them as continuous implies built-up is
"between" cropland and bare ground, which is meaningless, and lets a single
split straddle unrelated classes.

**Note on class sparsity.** LULC 50 (built-up) has a 46.7% landslide rate on
just 45 samples, and LULC 60 (bare/sparse) 7 positives of 12. These almost
certainly reflect genuine road-cut and denuded-slope failures, but the counts
are small enough to overfit badly. Target-statistic encoding with ordered
boosting mitigates this; the residual risk is recorded as a limitation and
revisited in Stage 4.

---

## 4. Evaluation

### 4.1 Spatial block cross-validation

**Decision.** Spatial block CV is the evaluation of record, at **10 km
blocks**. Random k-fold is run alongside purely to quantify the optimism it
injects, and is never used for model selection.

**Why.** Landslides cluster (§2.7). Under random splitting, a test point can
sit metres from a training point on the same slope, so the model is scored on
its ability to recall neighbours rather than to generalise to unseen terrain.
The reported inflation is expected to be in the 0.03–0.05 AUC range based on
the previous version's leakage analysis; the measured value goes in
[`results.md`](results.md).

**Measured result.** Random CV: ROC-AUC 0.9132, AUC-PR 0.7212. Spatial block
CV at 10 km: ROC-AUC 0.8694, AUC-PR 0.6564. **The random split inflates ROC-AUC
by 0.0438 and AUC-PR by 0.0648**, independently reproducing the previous phase's
0.03–0.05 estimate.

**Why block size was swept rather than taken from the variogram.** The original
plan was to derive one block size from the empirical autocorrelation range. That
plan was revised once the variogram was measured: it reaches 87% of the sill in
the first 500 m lag and is effectively flat thereafter, giving a 95%-of-sill
range of 2,500 m. That short range is partly an artefact of the sampling design
— negatives were drawn with a ≥100 m exclusion buffer and spread district-wide —
so a single number derived from it would have been a weak basis for a decision
that materially changes the headline result.

Sweeping instead exposed the mechanism directly:

| Block size | Points/block | Test points within 500 m of a training point | AUC-PR |
|---|---|---|---|
| 1 km | 1.3 | 11.2% | 0.7505 ± 0.1586 |
| 2.5 km | 3.1 | 4.8% | 0.7131 ± 0.1298 |
| 5 km | 10.9 | 3.3% | 0.6905 ± 0.0778 |
| **10 km** | 38.9 | 1.7% | **0.6564 ± 0.1179** |
| 20 km | 120.4 | 1.0% | 0.6567 ± 0.1261 |

At 1 km there are 1.3 points per block, so blocking is barely distinguishable
from random assignment and the resulting 0.7505 — above the random-CV figure,
with a ±0.1586 spread — is noise rather than a result. The score stabilises
between 10 km and 20 km, so 10 km was selected as the smallest block size at
which further separation buys nothing.

**Why report both schemes.** The inflated number is what a naive pipeline would
publish. Showing both makes the gap the finding.

### 4.2 Class imbalance

**Decision: no resampling and no reweighting. The imbalance is handled at the
threshold instead.**

Measured under spatial block CV at 10 km:

| Strategy | AUC-PR | ROC-AUC | Recall | Kappa |
|---|---|---|---|---|
| **None** | **0.6564** | 0.8694 | 0.5060 | 0.5875 |
| Class weights | 0.6450 | 0.8712 | 0.5875 | 0.6010 |
| `scale_pos_weight` | 0.6450 | 0.8712 | 0.5875 | 0.6010 |
| SMOTE (in-fold) | 0.6096 | 0.8676 | 0.5707 | 0.5704 |

**The prior expectation held, and is worth recording as a confirmed
prediction:** SMOTE was worst by 0.047 AUC-PR. Synthetic minority points
interpolate in *feature* space, and a synthetic sample between two real
landslides on different slopes need not correspond to any real terrain.

The less obvious result is that reweighting also slightly *hurt* AUC-PR while
improving recall and kappa. That is consistent with what reweighting actually
does: it shifts the operating point rather than improving the ranking. Since
this project sets its operating point explicitly and with a stated cost ratio
(§4.4), buying the same shift through the loss function would be doing the same
job twice, less transparently, at a small cost in ranking quality.

`class_weight` and `scale_pos_weight` coincide exactly for CatBoost — both
resolve to the same balanced reweighting — so they are one result, not two.

Any resampling happens strictly inside the fold; resampling before splitting
would place synthetic copies of training points into the test set.

### 4.3 Metrics

**Decision.** AUC-PR (average precision) is the primary metric. ROC-AUC, F1,
recall, precision and Cohen's kappa are reported alongside.

**Why.** At 1:10.8 imbalance, ROC-AUC is dominated by the majority class: a
model can post a strong ROC curve while having no operating point with usable
precision. The precision–recall curve is sensitive to exactly the regime that
matters. Both are reported because ROC-AUC is what the previous version
reported and comparability matters.

### 4.4 Operating threshold

**Decision: 0.1374**, minimising expected cost at a stated 10:1 ratio.

| Method | Threshold | Recall | Precision | Flagged |
|---|---|---|---|---|
| **Cost-minimising (10:1)** | **0.1374** | **0.650** | 0.559 | 9.8% |
| F2-optimal | 0.1382 | 0.650 | 0.560 | 9.8% |
| Youden J | 0.1382 | 0.650 | 0.560 | 9.8% |
| Precision floor (0.25) | 0.0687 | 0.782 | 0.250 | 26.4% |
| *0.5 (reference)* | 0.500 | **0.338** | 0.849 | 3.4% |

Three independent criteria converge on 0.137–0.138, so the operating point is
not an artefact of one objective.

**This is the single largest accuracy lever in the project.** Choosing the
threshold rather than defaulting to 0.5 takes recall from 0.338 to 0.650 —
roughly thirty times the effect of any architectural decision made elsewhere.
At 0.5 the model misses two-thirds of known landslide locations while reporting
85% precision, which is exactly the failure mode that makes a susceptibility map
dangerous rather than merely unhelpful.

Sensitivity to the assumed ratio is reported because 10:1 is an assumption, not
a measurement: the threshold is stable from 2:1 to 10:1 (0.246 → 0.137) and then
moves sharply, reaching 0.0375 at 50:1 where the model attains 97% recall by
flagging 74.7% of the district — useless to a planner.

**Why not 0.5.** 0.5 is the decision boundary of a balanced problem. This
problem is not balanced and its costs are not symmetric: a missed landslide
zone risks lives and a highway, while a false alarm costs a survey visit. The
10:1 ratio is a stated assumption rather than a measured one, so the sensitivity
of the resulting threshold to that ratio is reported too.

---

## 5. Ablation

**Result: the full-feature model alone has the best AUC-PR, and the commitment
made in advance is honoured here.**

| Rule | AUC-PR | ROC-AUC | Confidence? |
|---|---|---|---|
| **Full model alone** | **0.6480** | 0.8677 | **No** |
| **Full + thematic mean** *(adopted)* | 0.6456 | **0.8705** | **Yes** |
| Thematic weighted mean | 0.6050 | 0.8608 | Yes |
| Thematic meta-learner (logistic) | 0.5955 | 0.8418 | Yes |
| Thematic mean | 0.5697 | 0.8490 | Yes |
| Thematic noisy-OR | 0.5642 | 0.8476 | Yes |
| Thematic max | 0.5177 | 0.8321 | Yes |
| Thematic rank average | 0.5124 | 0.8501 | Yes |

**Decision: adopt `full_plus_thematic_mean` despite it not having the top
AUC-PR.** The reasoning, per §0.1:

- The deficit is **0.0024 AUC-PR**, against a fold-to-fold standard deviation of
  ±0.118 — about 2% of one standard deviation, well inside noise.
- It has the **best ROC-AUC in the table** (0.8705 vs 0.8677).
- It produces the confidence raster, which the single model structurally cannot.

The comparison that matters: the previous architecture's stack lost to its own
best member by **0.0104 and returned nothing for it**. This ensemble gives up
**0.0024 and returns the decision product the project exists to deliver**. That
is the difference between an unjustified ensemble and a justified one.

Note also that the logistic meta-learner — the previous design's combiner —
again placed mid-table (0.5955), well below a simple blend. Replacing the
meta-learner was never the fix; the composition was.

### 5b. Mixed learners per theme — tested, measured, not adopted

Letting each theme choose its own algorithm is a genuinely different proposal
from the one that failed before. The old design varied the *algorithm* while
every learner saw *identical features*; this varies both, adding a second
diversity axis rather than substituting for the first. It was therefore tested
rather than dismissed.

Per-theme best learners: topographic → CatBoost, hydrological → **logistic**,
geological → **logistic**, anthropogenic → CatBoost. On 2–4 features with 417
positives, boosted trees overfit where a regularised linear model does not.

| Combination | Uniform | Mixed | Δ |
|---|---|---|---|
| Thematic mean | 0.5697 | **0.6054** | +0.0357 |
| Thematic weighted mean | 0.6050 | 0.6139 | +0.0089 |
| Full + thematic mean | 0.6456 | **0.6557** | +0.0101 |

**The mixed ensemble at 0.6557 is the best AUC-PR measured anywhere in this
study**, above the full model alone.

**Cost: mean cross-theme prediction correlation rises 0.181 → 0.274, a 51%
increase.**

**Decision: keep the uniform learner; document the mixed variant as a validated
alternative.** The gain is 8.6% of one fold standard deviation. The cost falls
entirely on the disagreement signal — with mixed learners, part of any
disagreement reflects "a boosted tree and a linear model behave differently"
rather than "the physical evidence conflicts", and that signal is the one output
the full-feature model cannot produce at all. Trading half the decorrelation for
a tenth of a fold's noise is the wrong trade *for this project's objective*.

Both numbers are published so a reader weighting accuracy above decision quality
can choose otherwise; switching is a one-line configuration change. There is a
secondary methodological reason too: holding the algorithm fixed is the control
that makes "diversity from data views works" a testable claim rather than a
result confounded across two simultaneous changes.

---

## 6. Confidence from disagreement

**Decision made in advance, because it changes Stage 4's outputs:** each
thematic model is **calibrated on spatially-blocked out-of-fold predictions
before** any disagreement measure is computed.

**Why.** Variance across uncalibrated models measures how differently they are
*scaled*, not how differently they *judge*. A model that is systematically
confident and a model that is systematically conservative will show high
"disagreement" everywhere, including where they rank pixels identically. Since
the confidence raster is the actual product of this architecture, computing it
from uncalibrated scores would make the headline deliverable an artefact of
calibration mismatch.

**Measured: the precaution was justified.** Mean cross-model disagreement fell
from **0.0629 uncalibrated to 0.0522 calibrated** — **17% of the apparent
disagreement was calibration mismatch, not disagreement about the pixel.** The
two measures correlate at 0.938, so the ranking largely survives, but a sixth of
the magnitude of the project's headline output would have been an artefact of
how each model happened to be scaled.

All four thematic models were under-predicting the positive rate before
calibration (mean p 0.0655–0.0833 against an observed base rate of 0.0845); all
four now sit within 0.0015 of it.

**An unexpected result worth recording.** In the out-of-fold four-quadrant
breakdown, **263 of 417 known landslides (63%) fall in the high-susceptibility /
low-confidence quadrant** — "survey first" — against only 8 in
high-susceptibility / high-confidence.

This was not predicted, and it is a finding rather than a defect. It says that
real failure sites in this district are typically *not* places where all four
physical themes are adverse at once; they are places where one dominant
mechanism is present and the other lines of evidence are unremarkable. That is
precisely the population a survey team should be dispatched to, and a
single-model map would have rendered these as ordinary high-susceptibility
pixels with no signal that the evidence behind them was thin.

---

## 7. Decision layer

### 7.1 Rainfall intensity–duration thresholds

**Finding.** No rainfall time series exists in the delivered assets — only
`rainfall_uk_5yr_avg.tif`, a climatological average. An intensity–duration
threshold cannot be calibrated from a static mean.

**Decision.** Implement the component as a parameterised
`I = α · D^(−β)` model seeded with published regional values, and label it
**uncalibrated for Uttara Kannada** in the configuration, the code, and every
output it touches.

**Why.** The component is structurally real and the pipeline slot is live, so
supplying a rainfall time series later calibrates it without any code change.
Labelling it uncalibrated is what keeps it from being read as a validated
local result.

**Rejected: derive a threshold from the 5-year average raster.** Would produce
a number that looks like a calibrated threshold and is not one. This is the
single most tempting way to make the deliverable look more complete than it is.

**Rejected: drop the component.** Loses a required part of the chain and
removes the slot that makes later calibration cheap.

### 7.2 Exposure

**Decision.** Exposure is built from the road network and LULC built-up
(class 50), weighted 0.6 / 0.4.

**Why.** These are the only exposure layers present in the delivered assets. No
census settlement layer exists.

**Limitation recorded.** "Settlement" here means built-up *land cover* at 30 m,
not buildings or population. Individual houses below the 30 m cell are
invisible, and the weighting is a stated assumption, not a calibrated one.

### 7.3 NISAR as a post-hoc multiplier

**Decision.** Ground deformation enters after inference as an activity
multiplier, never as a training feature. Controlled by `decision.nisar` in the
configuration; disabled until a raster is supplied.

**Why.** Three reasons, in order of severity:

1. **Temporal leakage.** NISAR observations postdate the GSI inventory. A model
   trained on them would be using post-event deformation to "predict" events
   that already happened.
2. **Retraining cost.** As a feature, every new acquisition invalidates the
   model. As a multiplier, new data is a configuration change.
3. **Coherence.** L-band helps under canopy, but Uttara Kannada is densely
   forested and coherence will still be uneven. A feature with spatially
   structured reliability is far more damaging than a multiplier with the same
   property, because its unreliability propagates into every fitted parameter.

The separation is also conceptually right: susceptibility answers *where the
terrain is predisposed*; deformation answers *where something is moving now*.
Multiplying them gives intervention priority, and keeping them separate means
each can be updated, audited and disbelieved independently.

---

## 8. Limitations

Maintained as findings accumulate. Nothing here is resolved.

### Inventory

- **Completeness is unknown.** GSI records are reporting-driven: failures near
  roads and settlements are far likelier to be recorded than failures in
  forest interior. `dist_road` is therefore partly a *reporting* covariate as
  well as a causal one, and its SHAP importance should be read with that in
  mind.
- **417 positives is a small positive class** for a 10,142 km² district.

### Sampling

- **The ≥101 m negative buffer is learnable** (§2.7). The model can partly
  infer "distance from a known positive", which does not transfer to unmapped
  terrain.
- Negative points are pseudo-absences. Absence of a recorded landslide is not
  evidence of stability.

### Data coverage

- **The coastal GLiM gap** (§2.1) confounds the geological theme.
- **LULC classes 50 and 60 are sparse** (45 and 12 samples) with extreme
  landslide rates.

### Temporal

- **Static predictors, multi-year events.** NDVI, LULC and the 5-year rainfall
  average are single snapshots, while the inventory spans years. A slope that
  was forested when it failed may be mapped as bare today, and vice versa.
- **No event dates are used**, so nothing in the model is time-aware.

### Resolution

- **30.46 m cells.** Many Western Ghats failures are narrower than one cell,
  particularly road-cut failures along the ghat sections. The map identifies
  susceptible *zones*, not individual slopes.
- The DEM is stored as `int16`, so elevation is quantised to 1 m, and derived
  curvature inherits that quantisation.

### Unvalidated components

- **The rainfall threshold is uncalibrated** (§7.1).
- **Exposure weights are assumed, not fitted** (§7.2).
- **The NISAR multiplier is untested** — no deformation raster has been
  supplied.
- **No independent temporal holdout.** There is no post-inventory event set to
  validate against, so all reported performance is cross-validated rather than
  prospectively verified.
