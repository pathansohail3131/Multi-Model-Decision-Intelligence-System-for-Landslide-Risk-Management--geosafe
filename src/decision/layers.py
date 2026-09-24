"""Stage 8: the decision layers above susceptibility.

A probability per pixel is not something a planner can act on. This module
chains the components that turn a susceptibility surface into a prioritised
allocation of survey and mitigation effort.

Every component here is honest about its own status, because two of the three
are structurally real but not locally calibrated:

* **Rainfall intensity-duration thresholds.** No rainfall time series exists in
  the delivered assets — only a 5-year climatological mean — so the ID
  threshold cannot be calibrated for this district. It is implemented as a
  parameterised component seeded with published regional values and is labelled
  uncalibrated in the configuration, in the code, and in every output it
  touches. Deriving a "threshold" from a static mean would produce a number
  that looks calibrated and is not, which is the single most tempting way to
  make this deliverable look more complete than it is.

* **Exposure.** Roads and built-up land cover are the only exposure layers
  present. There is no census settlement layer, no population raster and no
  critical-infrastructure inventory, so "settlement" means built-up land cover
  at 30 m and exposure is not weighted by how many people are actually there.

* **NISAR ground deformation.** A post-hoc activity multiplier slot, disabled
  until a deformation raster is supplied. Deliberately not a training feature:
  NISAR observations postdate the GSI inventory, so using them as a predictor
  would mean using post-event deformation to "predict" events that already
  happened, and would force a retrain on every new acquisition.
"""

from __future__ import annotations

import json

import numpy as np
import rasterio
from rasterio.features import rasterize

from src.utils.logging_setup import get_logger

__all__ = [
    "intensity_duration_threshold",
    "rainfall_scenarios",
    "build_exposure_raster",
    "run_decision_stage",
    "QUADRANT_INTERPRETATION",
]

logger = get_logger(__name__)

#: What each cell of the planner's susceptibility x confidence table means.
QUADRANT_INTERPRETATION = {
    "act_now": "High susceptibility, themes agree — commit mitigation.",
    "survey_first": "High susceptibility, themes disagree — send a survey team before committing.",
    "monitor": "Low susceptibility but uncertain — keep under review.",
    "deprioritise": "Low susceptibility, themes agree — lowest call on resources.",
}


# ---------------------------------------------------------------------------
# rainfall intensity-duration
# ---------------------------------------------------------------------------
def intensity_duration_threshold(
    duration_h: np.ndarray | float, alpha: float = 15.0, beta: float = 0.44
) -> np.ndarray:
    """Rainfall intensity at which landsliding becomes likely, per duration.

    Implements the standard Caine-form power law ``I = alpha * D**(-beta)``,
    where *I* is mean rainfall intensity in mm/h and *D* is duration in hours.
    The negative exponent encodes the physical fact that a short, violent burst
    and a long, gentle soaking can both trigger failure: less intensity is
    needed as duration grows.

    Parameters
    ----------
    duration_h : array-like or float
        Storm duration in hours.
    alpha : float, default 15.0
        Scale parameter, in mm/h at one hour.
    beta : float, default 0.44
        Decay exponent.

    Returns
    -------
    numpy.ndarray
        Threshold intensity in mm/h.

    Warnings
    --------
    **Uncalibrated for Uttara Kannada.** The defaults are published regional
    values, not values fitted to this district's rainfall and inventory. A
    local calibration needs a rainfall time series and dated landslide events;
    neither is present in the delivered assets. Supplying a time series later
    calibrates this component with no change to any other part of the
    pipeline — the slot is live, only the numbers are borrowed.
    """
    duration = np.asarray(duration_h, dtype="float64")
    return alpha * np.power(duration, -beta)


def rainfall_scenarios(cfg) -> dict:
    """Build the rainfall trigger table across the configured durations.

    Returns
    -------
    dict
        Threshold intensity and total accumulation for each duration, plus an
        explicit calibration-status flag that downstream consumers and readers
        cannot miss.
    """
    params = cfg.get("decision", "rainfall", default={})
    alpha = float(params.get("alpha", 15.0))
    beta = float(params.get("beta", 0.44))
    durations = list(params.get("durations_h", [6, 12, 24, 48, 72]))

    table = {}
    for duration in durations:
        intensity = float(intensity_duration_threshold(duration, alpha, beta))
        table[f"{int(duration)}h"] = {
            "duration_h": int(duration),
            "threshold_intensity_mm_per_h": round(intensity, 3),
            "threshold_accumulation_mm": round(intensity * duration, 1),
        }

    logger.info("Rainfall ID thresholds (UNCALIBRATED, published parameters):")
    for key, row in table.items():
        logger.info(
            "  %-5s  %.2f mm/h  -> %.0f mm total",
            key,
            row["threshold_intensity_mm_per_h"],
            row["threshold_accumulation_mm"],
        )

    return {
        "model": "I = alpha * D^(-beta)",
        "alpha": alpha,
        "beta": beta,
        "calibrated_for_district": bool(params.get("calibrated", False)),
        "calibration_status": (
            "UNCALIBRATED — published regional parameters. No rainfall time "
            "series exists in the delivered assets (only a 5-year mean raster), "
            "so no local calibration is possible. Do not quote these as "
            "Uttara Kannada thresholds."
        ),
        "thresholds": table,
    }


# ---------------------------------------------------------------------------
# exposure
# ---------------------------------------------------------------------------
def build_exposure_raster(cfg, reference_path) -> tuple[np.ndarray, dict]:
    """Build a 0-1 exposure surface from roads and built-up land cover.

    Exposure is deliberately *not* multiplied into susceptibility inside the
    model. Susceptibility answers "is this slope predisposed to fail"; exposure
    answers "would anyone be affected if it did". Keeping them separate means
    each can be updated, audited and disbelieved independently, and a planner
    can see which of the two is driving a priority score.

    Parameters
    ----------
    cfg : Config
        Resolved project configuration.
    reference_path : Path
        Raster defining the output grid.

    Returns
    -------
    exposure : numpy.ndarray
        Exposure in [0, 1] on the reference grid.
    info : dict
        What went into it and with what weights.
    """
    params = cfg.get("decision", "exposure", default={})
    weights = params.get("weights", {"roads": 0.6, "lulc_builtup": 0.4})
    buffers = list(params.get("road_buffer_m", [100, 250, 500]))
    builtup_class = float(params.get("lulc_builtup_class", 50))

    with rasterio.open(reference_path) as ref:
        transform = ref.transform
        shape = (ref.height, ref.width)
        resolution = ref.res[0]

    info: dict = {"components": {}, "weights": dict(weights), "road_buffers_m": buffers}

    # --- roads: graded proximity bands rather than a single buffer ---------
    road_component = np.zeros(shape, dtype="float32")
    roads_path = cfg.roads_shapefile
    if roads_path.exists():
        import geopandas as gpd

        roads = gpd.read_file(roads_path)
        with rasterio.open(reference_path) as ref:
            if roads.crs is not None and str(roads.crs) != str(ref.crs):
                # Reproject explicitly rather than relying on a silent match.
                logger.info("Reprojecting roads from %s to %s", roads.crs, ref.crs)
                roads = roads.to_crs(ref.crs)

        # Nearest band wins, so scores decay outward: 1.0, 0.66, 0.33.
        for rank, buffer_m in enumerate(sorted(buffers)):
            score = 1.0 - rank / max(len(buffers), 1)
            geometries = [
                (geom.buffer(buffer_m), score)
                for geom in roads.geometry
                if geom is not None
            ]
            if not geometries:
                continue
            band = rasterize(
                geometries,
                out_shape=shape,
                transform=transform,
                fill=0.0,
                dtype="float32",
                merge_alg=rasterio.enums.MergeAlg.replace,
            )
            road_component = np.maximum(road_component, band)
        info["components"]["roads"] = {
            "source": str(roads_path.name),
            "n_features": int(len(roads)),
            "covered_pixels": int((road_component > 0).sum()),
        }
    else:
        logger.warning("roads shapefile not found at %s; roads exposure is zero", roads_path)
        info["components"]["roads"] = {"source": None, "covered_pixels": 0}

    # --- built-up land cover ------------------------------------------------
    builtup_component = np.zeros(shape, dtype="float32")
    lulc_path = cfg.raster_path("LULC")
    if lulc_path.exists():
        with rasterio.open(lulc_path) as src:
            lulc = src.read(1).astype("float64")
            nodata = src.nodata
        ok = np.isfinite(lulc)
        if nodata is not None:
            ok &= ~np.isclose(lulc, nodata)
        builtup_component[ok & np.isclose(lulc, builtup_class)] = 1.0
        info["components"]["lulc_builtup"] = {
            "source": lulc_path.name,
            "class_code": builtup_class,
            "covered_pixels": int((builtup_component > 0).sum()),
        }

    exposure = (
        float(weights.get("roads", 0.6)) * road_component
        + float(weights.get("lulc_builtup", 0.4)) * builtup_component
    )
    exposure = np.clip(exposure, 0.0, 1.0).astype("float32")

    info["limitation"] = (
        "Exposure is built from road proximity and built-up LAND COVER at 30 m. "
        "There is no census settlement layer, no population raster and no "
        "critical-infrastructure inventory in the delivered assets. Buildings "
        "smaller than one cell are invisible, and exposure is not weighted by "
        "population. The 0.6/0.4 weighting is a stated assumption, not a "
        "calibrated one."
    )
    logger.info(
        "Exposure built: %d road-proximal px, %d built-up px",
        info["components"].get("roads", {}).get("covered_pixels", 0),
        info["components"].get("lulc_builtup", {}).get("covered_pixels", 0),
    )
    return exposure, info


# ---------------------------------------------------------------------------
# prioritisation
# ---------------------------------------------------------------------------
def run_decision_stage(cfg) -> dict:
    """Stage 8 entry point: exposure overlay, priority surface, NISAR slot.

    Parameters
    ----------
    cfg : Config
        Resolved project configuration.

    Returns
    -------
    dict
        Report written to ``outputs/reports/decision.json``.
    """
    # Prefer the complete district map; fall back to a partial one so the
    # decision layer can still be exercised and reviewed. The suffix is
    # carried through to every output, so a partial result can never be
    # mistaken for the deliverable further down the line.
    susceptibility_path = cfg.rasters_dir / "susceptibility.tif"
    suffix = ""
    if not susceptibility_path.exists():
        candidates = sorted(cfg.rasters_dir.glob("susceptibility_partial_*.tif"))
        if not candidates:
            raise FileNotFoundError(
                f"{susceptibility_path} not found; run stage 7 before stage 8"
            )
        susceptibility_path = candidates[0]
        suffix = susceptibility_path.stem.replace("susceptibility", "")
        logger.warning(
            "Using PARTIAL susceptibility raster %s. Decision outputs are "
            "provisional and carry the same suffix.",
            susceptibility_path.name,
        )
    confidence_path = cfg.rasters_dir / f"confidence{suffix}.tif"

    report: dict = {"rainfall": rainfall_scenarios(cfg)}

    exposure, exposure_info = build_exposure_raster(cfg, susceptibility_path)
    report["exposure"] = exposure_info

    with rasterio.open(susceptibility_path) as src:
        susceptibility = src.read(1)
        profile = src.profile.copy()
        nodata = src.nodata
        res_x, res_y = src.res
    with rasterio.open(confidence_path) as src:
        confidence = src.read(1)

    valid = (
        np.isfinite(susceptibility)
        & ~np.isclose(susceptibility, nodata)
        & np.isfinite(confidence)
        & ~np.isclose(confidence, nodata)
    )

    # --- NISAR post-hoc activity multiplier -------------------------------
    nisar_cfg = cfg.get("decision", "nisar", default={})
    multiplier = np.ones_like(susceptibility, dtype="float32")
    nisar_info = {"enabled": False, "note": "no deformation raster supplied"}
    if nisar_cfg.get("enabled") and nisar_cfg.get("raster"):
        path = cfg.data_path(str(nisar_cfg["raster"]))
        if path.exists():
            low, high = nisar_cfg.get("multiplier_range", [1.0, 1.5])
            with rasterio.open(path) as src:
                deformation = src.read(1).astype("float32")
                dn = src.nodata
            ok = np.isfinite(deformation)
            if dn is not None:
                ok &= ~np.isclose(deformation, dn)
            if ok.any():
                lo, hi = np.nanpercentile(deformation[ok], [5, 95])
                scaled = np.clip((deformation - lo) / max(hi - lo, 1e-9), 0, 1)
                multiplier[ok] = (low + (high - low) * scaled[ok]).astype("float32")
            nisar_info = {
                "enabled": True,
                "raster": str(path),
                "multiplier_range": [float(low), float(high)],
                "note": (
                    "Applied after inference as an activity multiplier. No "
                    "model was retrained."
                ),
            }
        else:
            nisar_info["note"] = f"configured raster not found: {path}"
    report["nisar"] = nisar_info

    # --- priority surface ---------------------------------------------------
    # Susceptibility x confidence x exposure, times the activity multiplier.
    # Confidence enters as a multiplier rather than a filter so that an
    # uncertain-but-severe location is demoted rather than discarded — it still
    # appears, lower down, as a survey candidate.
    priority = np.full_like(susceptibility, nodata, dtype="float32")
    priority[valid] = (
        susceptibility[valid]
        * confidence[valid]
        * np.clip(exposure[valid], 0.05, 1.0)
        * multiplier[valid]
    ).astype("float32")

    priority_path = cfg.rasters_dir / f"priority{suffix}.tif"
    with rasterio.open(priority_path, "w", **profile) as dst:
        dst.write(priority, 1)

    exposure_path = cfg.rasters_dir / f"exposure{suffix}.tif"
    with rasterio.open(exposure_path, "w", **profile) as dst:
        out = np.full_like(exposure, nodata, dtype="float32")
        out[valid] = exposure[valid]
        dst.write(out, 1)

    # --- the four-quadrant planner table -----------------------------------
    threshold = float(
        json.loads((cfg.reports_dir / f"inference{suffix}.json").read_text())["threshold"]
    )
    conf_split = float(np.nanmedian(confidence[valid]))
    cell_km2 = res_x * res_y / 1e6

    high_susc = valid & (susceptibility >= threshold)
    high_conf = valid & (confidence >= conf_split)
    quadrants = {
        "act_now": int((high_susc & high_conf).sum()),
        "survey_first": int((high_susc & ~high_conf).sum()),
        "monitor": int((~high_susc & ~high_conf & valid).sum()),
        "deprioritise": int((~high_susc & high_conf & valid).sum()),
    }
    report["decision_quadrants"] = {
        "susceptibility_threshold": threshold,
        "confidence_split": round(conf_split, 4),
        "pixels": quadrants,
        "area_km2": {k: round(v * cell_km2, 1) for k, v in quadrants.items()},
        "interpretation": dict(QUADRANT_INTERPRETATION),
    }
    logger.info("Planner quadrants (km2):")
    for key, value in report["decision_quadrants"]["area_km2"].items():
        logger.info("  %-14s %8.1f km2", key, value)

    # --- top priority zones -------------------------------------------------
    ok_priority = priority[valid]
    cut = float(np.nanpercentile(ok_priority, 99))
    report["priority_surface"] = {
        "path": str(priority_path),
        "exposure_path": str(exposure_path),
        "formula": "susceptibility x confidence x exposure x nisar_multiplier",
        "p50": round(float(np.nanpercentile(ok_priority, 50)), 5),
        "p90": round(float(np.nanpercentile(ok_priority, 90)), 5),
        "p99": round(cut, 5),
        "top_1pct_area_km2": round(float((ok_priority >= cut).sum()) * cell_km2, 1),
    }
    logger.info(
        "Top 1%% priority zones cover %.1f km2 — the shortlist a planner works from",
        report["priority_surface"]["top_1pct_area_km2"],
    )

    report["is_partial"] = bool(suffix)
    out_path = cfg.reports_dir / f"decision{suffix}.json"
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)
    logger.info("Decision report written: %s", out_path)

    return report
