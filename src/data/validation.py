"""Validating that the delivered rasters are what they claim to be.

The previous phase asserted that all 14 conditioning-factor rasters share a
common grid, CRS and extent. That claim is checked here rather than trusted,
because misalignment is the single most destructive silent failure in raster
machine learning: every stage downstream completes normally, a plausible map is
written, and every pixel has sampled the wrong location.

Validation returns a report rather than raising on the first problem, so that
one run reveals the full picture instead of revealing it one failure at a time.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import rasterio

from src.data.loaders import RasterMeta, all_raster_metadata
from src.utils.logging_setup import get_logger

__all__ = ["GridReport", "validate_grid_alignment", "district_mask_stats"]

logger = get_logger(__name__)


@dataclass
class GridReport:
    """Outcome of the raster-alignment check.

    Attributes
    ----------
    aligned : bool
        True when every raster shares one grid.
    reference : str or None
        Factor name used as the reference grid.
    groups : dict
        Grid signature mapped to the factors sharing it. More than one entry
        means misalignment.
    config_matches : bool
        Whether the shared grid matches the grid declared in ``config.yaml``.
    problems : list of str
        Human-readable problems.
    nodata_by_value : dict
        Nodata value mapped to the factors using it. Expected to have more than
        one entry in this project; recorded so the per-file handling is visible
        rather than assumed.
    metadata : dict
        Factor name mapped to its serialised metadata.
    """

    aligned: bool = False
    reference: str | None = None
    groups: dict = field(default_factory=dict)
    config_matches: bool = False
    problems: list[str] = field(default_factory=list)
    nodata_by_value: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        """Return a JSON-serialisable summary."""
        return {
            "aligned": self.aligned,
            "reference": self.reference,
            "n_grid_groups": len(self.groups),
            "grid_groups": {str(k): v for k, v in self.groups.items()},
            "config_matches": self.config_matches,
            "problems": self.problems,
            "nodata_by_value": {str(k): v for k, v in self.nodata_by_value.items()},
            "metadata": self.metadata,
        }


def validate_grid_alignment(cfg) -> GridReport:
    """Check that every conditioning-factor raster shares one grid.

    Two things are verified:

    1. **Mutual alignment** — all rasters have identical width, height, CRS and
       affine transform. Resolution and bounds follow from those, so they are
       not checked separately.
    2. **Agreement with configuration** — the shared grid matches the ``grid``
       block in ``config.yaml``. This catches the case where the data assets
       are internally consistent but are not the assets the project was
       configured against, for example after a partial re-derivation.

    Parameters
    ----------
    cfg : Config
        Resolved project configuration.

    Returns
    -------
    GridReport
        The validation outcome. Inspect ``aligned`` and ``problems``.
    """
    # Missing rasters are reported, not raised on: stages 1-6 work from the
    # training CSV alone, so an unavailable layer should degrade the report
    # rather than abort the audit.
    missing = cfg.missing_rasters()
    present = [f for f in cfg.raster_factors if f not in missing]
    metas: dict[str, RasterMeta] = all_raster_metadata(cfg, present)
    report = GridReport(metadata={k: v.as_dict() for k, v in metas.items()})

    if missing:
        msg = f"{len(missing)} raster(s) not on disk: {sorted(missing)}"
        report.problems.append(msg)
        logger.warning("Grid validation is partial — %s", msg)

    if not metas:
        report.problems.append("no rasters available; grid alignment not checked")
        logger.error("No rasters available to validate")
        return report

    # --- group rasters by grid signature ---------------------------------
    groups: dict[tuple, list[str]] = {}
    for name, meta in metas.items():
        groups.setdefault(meta.grid_key(), []).append(name)
    report.groups = {str(k): sorted(v) for k, v in groups.items()}

    if len(groups) == 1:
        report.aligned = True
        report.reference = sorted(metas)[0]
        logger.info("Grid alignment: OK — all %d rasters share one grid", len(metas))
    else:
        report.aligned = False
        msg = f"rasters do not share one grid: {len(groups)} distinct grids found"
        report.problems.append(msg)
        logger.error("Grid alignment: FAILED — %s", msg)
        for key, names in groups.items():
            logger.error("  grid %s: %s", key[:3], sorted(names))

    # --- compare against the configured grid ------------------------------
    declared = cfg.get("grid", default={})
    if declared and report.aligned:
        ref = metas[report.reference]
        checks = {
            "width": (ref.width, declared.get("width")),
            "height": (ref.height, declared.get("height")),
            "crs": (ref.crs, declared.get("crs")),
        }
        mismatches = [
            f"{field}: raster={actual!r} config={expected!r}"
            for field, (actual, expected) in checks.items()
            if expected is not None and actual != expected
        ]
        # Resolution is float; compare with tolerance rather than equality.
        declared_res = declared.get("resolution_m")
        if declared_res is not None and not np.isclose(
            ref.resolution[0], float(declared_res), rtol=0, atol=1e-6
        ):
            mismatches.append(
                f"resolution_m: raster={ref.resolution[0]!r} config={declared_res!r}"
            )
        if mismatches:
            report.config_matches = False
            report.problems.extend(mismatches)
            logger.error("Configured grid does not match the data: %s", mismatches)
        else:
            report.config_matches = True
            logger.info("Configured grid matches the delivered rasters")

    # --- record the nodata conventions in play ----------------------------
    for name, meta in metas.items():
        report.nodata_by_value.setdefault(meta.nodata, []).append(name)
    for value, names in sorted(
        report.nodata_by_value.items(), key=lambda kv: str(kv[0])
    ):
        logger.info("nodata %s used by %d layer(s): %s", value, len(names), sorted(names))
    if len(report.nodata_by_value) > 1:
        logger.info(
            "Multiple nodata conventions present — each raster's own header value "
            "is used at read time; no global nodata constant exists in this codebase."
        )

    if None in report.nodata_by_value:
        report.problems.append(
            "layer(s) declare no nodata value: "
            f"{sorted(report.nodata_by_value[None])} — masking cannot be verified"
        )

    return report


def district_mask_stats(cfg, reference_factor: str = "slope") -> dict:
    """Measure the in-district footprint from a reference raster's nodata mask.

    Parameters
    ----------
    cfg : Config
        Resolved project configuration.
    reference_factor : str, default "slope"
        Factor whose valid-data mask defines the district footprint.

    Returns
    -------
    dict
        Pixel counts, implied area, and comparison against the configured
        expectation.

    Notes
    -----
    The implied area is a useful end-to-end sanity check on CRS and cell size
    together: if the CRS were wrong or the resolution misread, the computed
    area would miss the district's known extent by a wide margin rather than
    by a percent or two.
    """
    path = cfg.raster_path(reference_factor)
    with rasterio.open(path) as src:
        data = src.read(1)
        nodata = src.nodata
        res_x, res_y = src.res
    valid = np.isfinite(data)
    if nodata is not None:
        valid &= ~np.isclose(data, nodata)

    n_valid = int(valid.sum())
    area_km2 = n_valid * res_x * res_y / 1e6
    expected = cfg.get("grid", "expected_valid_pixels")

    stats = {
        "reference_factor": reference_factor,
        "total_pixels": int(data.size),
        "valid_pixels": n_valid,
        "valid_fraction": round(n_valid / data.size, 4),
        "implied_area_km2": round(area_km2, 1),
        "expected_valid_pixels": expected,
        "matches_expected": (expected is None or n_valid == int(expected)),
    }
    logger.info(
        "District footprint from %s: %d valid px (%.1f%%) = %.0f km2",
        reference_factor,
        n_valid,
        100 * n_valid / data.size,
        area_km2,
    )
    if expected is not None and n_valid != int(expected):
        logger.warning(
            "Valid-pixel count %d differs from the configured expectation %s",
            n_valid,
            expected,
        )
    return stats
