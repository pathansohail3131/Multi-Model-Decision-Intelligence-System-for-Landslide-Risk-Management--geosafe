# Data Dictionary

Every column and raster layer used by this project: source, units, resolution,
CRS, derivation and known limitations.

**Status: Stage 0 of 8.** Values marked **verified** were measured directly
from the delivered assets on 2026-09-13. Values marked *inherited* come from
the previous phase's documentation and are re-verified programmatically in
Stage 1 (`python run_pipeline.py stage 1`). Anything still marked *unverified*
should not be quoted.

---

## 1. Common grid

All 14 conditioning-factor rasters share one grid. **Verified** by reading the
headers rather than trusting the claim:

| Property | Value |
|---|---|
| CRS | EPSG:32643 (WGS 84 / UTM zone 43N) |
| Width × height | 3,738 × 5,554 px |
| Total pixels | 20,760,852 |
| In-district (non-nodata) pixels | 10,929,202 (52.6%) |
| Cell size | 30.46253464981975 m (both axes) |
| Bounds (L, B, R, T) | 397835.3986, 1539148.5, 511704.35, 1708337.4146 |
| Implied area | ≈ 10,142 km² |
| Official district area | 10,291 km² |

The 1.4% shortfall against the official figure is consistent with coastline and
boundary clipping and is not treated as an error.

The odd cell size is the signature of 1-arcsecond SRTM data reprojected to
UTM 43N; it is not a rounding artefact and must not be "tidied" to 30 m,
because doing so would resample every layer and break the alignment.

### Nodata is not uniform

| Convention | Value | Layers |
|---|---|---|
| GDAL-derived | −9999 | elevation, slope, aspect, plan/profile curvature, dist_road, NDVI, LULC, lithology, rainfall |
| WhiteboxTools-derived | −32768 | TWI, SPI, dist_drainage, dist_fault |

Each raster's nodata is read from its own header at load time. No global nodata
constant exists in the codebase. Assuming one would admit −32768 into the
feature matrix as a real measurement, producing a map that looks correct and is
not.

---

## 2. Training dataset

`data/final/training_dataset_final.csv` — **verified**: 4,940 rows × 20
columns, zero missing values in any feature column.

### 2.1 Provenance

*Inherited, re-verified in Stage 1.*

**Positive samples.** Geological Survey of India field-validated landslide
records, obtained from the GSI Bhusanket portal ("Landslide Inventory (Field
Validated)").

| Step | Count |
|---|---|
| Raw Uttara Kannada records extracted | 523 |
| After taluk-level cross-verification | 454 |
| After boundary / raster-coverage cleanup | 452 |
| After removing 35 duplicate-coordinate records | **417** |

Two points fall ~900 m outside the district boundary and were retained, judged
to be boundary-precision rather than data error.

**Negative samples.** 4,536 points by slope-stratified sampling with a
buffer-exclusion zone around every landslide point, reduced to **4,523** by the
same cleanup. **Verified**: the minimum observed distance from any negative to
any positive is 101.2 m, consistent with a ≥100 m exclusion buffer.

**Verified**: the delivered CSV is the **pre-cleanup** file. It still contains
4 rows with physically impossible negative elevation (see §2.4) that the
previous phase dropped at load time. This project drops them explicitly in
Stage 1, giving 4,936 rows / 417 positive / 4,519 negative.

### 2.2 Columns

| Column | Type | Units | Role | Verified range / values |
|---|---|---|---|---|
| `elevation` | float | m above sea level | feature (topographic) | −3 to 978 |
| `slope` | float | degrees | feature (topographic) | 0 to 57.80 |
| `aspect` | float | degrees clockwise from north | feature (topographic), circular | 0 to 359.26 |
| `plan_curv` | float | 1/m | feature (topographic) | −0.9192 to 0.4178 |
| `profile_curv` | float | 1/m | feature (topographic) | −0.0138 to 0.0191 |
| `TWI` | float | dimensionless | feature (hydrological) | 3.661 to 24.857 |
| `SPI` | float | dimensionless | feature (hydrological) | 0.0004 to 2,250,643 |
| `dist_drainage` | float | m | feature (hydrological) | 0 to 2,999.7 |
| `dist_road` | float | m | feature (anthropogenic) | 0 to 6,861.9 |
| `dist_fault` | float | m | feature (geological) | 0 to 5,904.5 |
| `NDVI` | float | dimensionless, −1 to 1 | feature (anthropogenic) | −0.4475 to 0.9001 |
| `LULC` | float | class code | feature (anthropogenic), **categorical** | 10, 20, 30, 40, 50, 60, 80, 90, 95 |
| `lithology` | float | class code | feature (geological), **categorical** | −1, 1, 2, 3, 4, 5 |
| `rainfall` | float | mm/year | feature (hydrological) | 1,114.75 to 4,009.81 |
| `label` | int | 0 / 1 | **target** | 417 ones, 4,523 zeros |
| `X` | float | m (EPSG:32643 easting) | spatial blocking only | 402,816 to 510,440 |
| `Y` | float | m (EPSG:32643 northing) | spatial blocking only | 1,539,834 to 1,708,200 |
| `taluk` | str | — | **forbidden** — reporting only | 10 taluks; null on all negatives |
| `slide_name` | str | — | **forbidden** — reporting only | 173 distinct; null on all negatives |
| `sample_id` | int | — | identifier | 1 to 4,940 |

`X` and `Y` are listed in `features.forbidden_columns` alongside `taluk` and
`slide_name`. They are essential for spatial blocking and for the district
raster, but as model inputs they would let the learner memorise locations
instead of conditions.

### 2.3 Class balance

**Verified**: 417 positive / 4,523 negative = **1:10.85**.

Positives by taluk: Honavar 88, Siddapur 56, Supa 53, Sirsi 50, Kumta 49,
Karwar 42, Ankola 40, Yellapur 34, Dandeli 4, Mundgod 1.

The Mundgod and Dandeli counts are very low. Both are eastern, lower-relief
taluks, so this is expected rather than a sampling failure, but any
taluk-stratified reporting should treat those two as effectively unsampled.

### 2.4 Known data issues

All **verified** directly.

| # | Issue | Extent | Handling |
|---|---|---|---|
| 1 | `lithology = -1` — GLiM coverage gap, not in the class lookup | 157 rows (43 positive) | Explicit `unmapped_coastal_gap` category; see §4.2 and methodology §2.1 |
| 2 | Negative elevation — DEM interpolation artefact over water | 4 rows, all negatives (−1 to −3 m; 3 of 4 are LULC 80 = permanent water) | Dropped in Stage 1 |
| 3 | `aspect = 270.0` exactly wherever `slope = 0` — flat-cell placeholder | 61 rows | `is_flat` flag; aspect components zeroed |
| 4 | `SPI` skew 46.3 over four orders of magnitude | all rows | `log1p` → skew 1.29 |
| 5 | `taluk` / `slide_name` non-null only on positives | 4,523 nulls each | Blocked as features |
| 6 | Positives cluster spatially — 86 of 417 within 100 m of another positive (min 0.65 m) | — | Spatial block CV |
| 7 | Two rows share an identical feature vector | 2 rows | Reviewed in Stage 1 |
| 8 | LULC 80 (permanent water) carries 105 negatives and 0 positives | 105 rows | Open question for Stage 1 — water cannot fail, so these may be trivially-separable negatives that flatter the metrics |

---

## 3. Conditioning-factor rasters

All in `data/processed/`, all on the §1 grid.

### 3.1 Topographic

| Factor | File | Units | dtype | Derivation |
|---|---|---|---|---|
| elevation | `dem_uk_utm.tif` | m | int16 | SRTM 1-arcsec DEM, void-filled, reprojected to EPSG:32643 |
| slope | `slope_uk.tif` | degrees | float32 | GDAL slope from the filled DEM |
| aspect | `aspect_uk.tif` | degrees | float32 | GDAL aspect; flat cells emit 270.0 |
| plan curvature | `plan_curv_uk.tif` | 1/m | float32 | Curvature perpendicular to the slope direction — controls flow convergence |
| profile curvature | `profile_curv_uk.tif` | 1/m | float32 | Curvature along the slope direction — controls flow acceleration |

**Limitation.** `int16` elevation quantises height to 1 m. Curvature is a second
derivative of that surface, so it inherits the quantisation and is noisier than
its float storage implies.

**Limitation.** *Inherited*: nodata-edge slope artefacts were identified and
fixed in the previous phase. Stage 1 re-checks the district boundary for
residual edge effects.

### 3.2 Hydrological

| Factor | File | Units | dtype | Derivation |
|---|---|---|---|---|
| TWI | `TWI_uk.tif` | dimensionless | float32 | ln(specific catchment area / tan slope), WhiteboxTools |
| SPI | `SPI_uk.tif` | dimensionless | float32 | specific catchment area × tan slope, WhiteboxTools |
| distance to drainage | `dist_to_drainage.tif` | m | float32 | Euclidean distance to extracted stream network |
| rainfall | `rainfall_uk_5yr_avg_aligned.tif` | mm/year | float32 | **CHIRPS** 5-year mean, 2019–2023, via Google Earth Engine |

**Limitation — rainfall source.** *Inherited.* CHIRPS was used rather than IMD
gridded data because of portal access constraints. CHIRPS is known to
underestimate orographic rainfall in high-relief terrain, so absolute values are
likely conservative for the Ghats crest. The *spatial pattern* — relative wet
and dry zones across the district — remains usable, which is what a
susceptibility model consumes. Absolute magnitudes should not be quoted as
rainfall estimates.

**Limitation — static rainfall.** A 5-year mean carries no temporal
information, which is why the intensity–duration threshold component in Stage 8
cannot be locally calibrated (methodology §7.1).

**Ambiguity to resolve in Stage 1.** `processed/` contains both
`rainfall_uk_aligned.tif` and `rainfall_uk_5yr_avg_aligned.tif`. The
configuration uses the 5-year average per the brief; Stage 1 re-samples both at
the training coordinates to confirm which one the CSV was actually built from.

### 3.3 Geological

| Factor | File | Units | dtype | Derivation |
|---|---|---|---|---|
| lithology | `lithology_uk.tif` | class code | float32 | GLiM (Global Lithological Map) v1.1, rasterised |
| distance to fault | `dist_to_fault.tif` | m | float32 | Euclidean distance to **DEM-derived structural lineaments** |

**Lithology class codes** (`processed/lithology_classes.json`):

| Code | GLiM class | Meaning | Samples | Landslide rate |
|---|---|---|---|---|
| −1 | *(absent from lookup)* | Unmapped — coastal coverage gap | 157 | 27.4% |
| 1 | `mt` | Metamorphic | 3,146 | 8.5% |
| 2 | `pa` | Acid plutonic | 448 | 5.8% |
| 3 | `su` | Siliciclastic sedimentary | 876 | 7.2% |
| 4 | `vb` | Basic volcanic | 301 | 5.0% |
| 5 | `wb` | Water bodies | 12 | 25.0% |

Code 5 has only 12 samples; its 25% rate is three positives and is not
interpretable.

**Limitation — `dist_fault` is a proxy, not fault data.** *Inherited, and
important.* No field-validated fault or lineament map exists for this district
at usable resolution, so this layer is a **DEM-derived structural-lineament
proxy**. It is a defensible substitution — and one much of the published
literature makes implicitly by not specifying a lineament source — but it must
not be described as distance to mapped geological faults. Since the lineaments
are derived from the same DEM as slope and curvature, this factor is **not
fully independent of the topographic theme**, which partially weakens the
thematic blindness the architecture relies on. Recorded as a limitation and
quantified in Stage 4 by measuring the correlation between geological and
topographic model predictions.

**Limitation — coastal coverage gap.** GLiM polygon coverage is absent along
the coastal strip, producing the `-1` sentinel in §2.4. See methodology §2.1.

### 3.4 Anthropogenic / surface

| Factor | File | Units | dtype | Derivation |
|---|---|---|---|---|
| distance to road | `dist_to_road.tif` | m | float32 | Euclidean distance to OSM-derived road network (`processed/roads_uk.shp`) |
| LULC | `LULC_uk_aligned.tif` | class code | float32 | ESA WorldCover, resampled to the common grid |
| NDVI | `NDVI_uk_aligned.tif` | dimensionless | float32 | Satellite-derived NDVI composite, resampled to the common grid |

**LULC class codes** (ESA WorldCover convention):

| Code | Class | Samples | Positives | Landslide rate |
|---|---|---|---|---|
| 10 | Tree cover | 4,161 | 360 | 8.7% |
| 20 | Shrubland | 35 | 0 | 0% |
| 30 | Grassland | 282 | 25 | 8.9% |
| 40 | Cropland | 291 | 4 | 1.4% |
| 50 | Built-up | 45 | 21 | **46.7%** |
| 60 | Bare / sparse vegetation | 12 | 7 | **58.3%** |
| 80 | Permanent water | 105 | 0 | 0% |
| 90 | Herbaceous wetland | 1 | 0 | 0% |
| 95 | Mangroves | 8 | 0 | 0% |

**Limitation — sparse classes with extreme rates.** Classes 50 and 60 carry
very high landslide rates on 45 and 12 samples. These plausibly reflect genuine
road-cut and denuded-slope failures, but the counts are small enough to overfit
badly. Handled by categorical target statistics with ordered boosting; residual
risk recorded.

**Limitation — `dist_road` is partly a reporting covariate.** *Inherited, and
the most important caveat in this document.* The previous phase found
`dist_road` to be the dominant SHAP feature. That plausibly reflects two
overlapping effects that the available data cannot disentangle:

1. genuine slope destabilisation from road-cutting, a well-documented Western
   Ghats landslide trigger; and
2. **inventory accessibility bias** — field-validated databases systematically
   under-record landslides in remote, roadless terrain.

Any statement about `dist_road` importance must carry this caveat. It is a
recognised limitation across the landslide-susceptibility literature wherever
field-surveyed inventories are used in place of exhaustive remote-sensing
mapping.

**Limitation — temporal mismatch.** LULC and NDVI are single snapshots while
the inventory spans multiple years. A slope that was forested when it failed
may be mapped as bare today, and vice versa.

---

## 4. Supporting vector layers

| Layer | File | CRS | Use |
|---|---|---|---|
| District boundary | `raw/boundary/uttara_kannada_boundary.shp` | EPSG:32643 | Masking, map extent |
| Verified landslide inventory | `samples/uttara_kannada_landslides_verified.shp` / `.csv` | EPSG:4326 | Positive samples, validation figures |
| Negative sample points | `samples/negative_points.shp` | EPSG:32643 | Negative samples |
| Road network | `processed/roads_uk.shp` | EPSG:32643 | `dist_road` derivation; Stage 8 exposure overlay |

The inventory shapefile carries a different CRS (`.prj` is 145 bytes, consistent
with a geographic definition) from the working grid. Any use of it must
reproject explicitly; Stage 1 asserts this rather than relying on silent
handling.

---

## 5. Layers present but not used

`data/processed/` holds intermediate products from the derivation phase that
this project does not consume: `dem_filled.tif`, `dem_clipped_4326.tif`,
`hillshade_uk.tif`, `slope_wbt.tif`, `edges_uk.tif`, `flow_acc_sca.tif`,
`streams_uk.tif`, `streams_uk_clean.tif`, `lineaments_uk_clean.tif`,
`LULC_uk.tif`, `NDVI_uk.tif`, `rainfall_uk.tif`, `rainfall_uk_5yr_avg.tif`.

They are retained as provenance for the aligned layers. `hillshade_uk.tif` is
used for cartographic basemaps in Stage 8. The unaligned originals
(`LULC_uk.tif`, `NDVI_uk.tif`, `rainfall_uk*.tif`) are **not** interchangeable
with their `_aligned` counterparts and must never be substituted.

`data/raw/lithology/` holds the full GLiM geodatabase (>1 GB) and
`data/raw/landslide_inventory/` holds the source GSI PDF and a NASA COOLR
extract. The COOLR extract is not used: it contains too few Uttara Kannada
records to add to the GSI inventory.

---

## 6. Exposure data (Stage 8)

| Need | Available | Status |
|---|---|---|
| Roads | `processed/roads_uk.shp` (OSM-derived) | Available |
| Settlements | LULC class 50 (built-up) | **Proxy only** |
| Population | — | **Absent** |
| Critical infrastructure | — | **Absent** |

**Limitation.** There is no census settlement layer, no population raster and
no critical-infrastructure inventory. "Settlement" in the exposure overlay means
built-up *land cover* at 30 m resolution — individual buildings below the cell
size are invisible, and exposure is not weighted by how many people are
actually there. The roads/built-up weighting (0.6 / 0.4) is a stated assumption,
not a calibrated one.
