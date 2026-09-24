"""District locator: a real map of Uttara Kannada and real terrain at a point.

Clicking a location in the simulator samples the same 14 conditioning-factor
rasters that district inference reads, so a scenario built this way is the
actual feature vector of that 30 m cell — not a guess.
"""

from __future__ import annotations

import io
import math
import threading

import numpy as np
import rasterio
from rasterio.enums import Resampling

from api.app.schemas.factors import CATEGORICAL, FACTORS
from src.data.loaders import sample_raster_at_points
from src.utils.logging_setup import get_logger

__all__ = ["DistrictService", "CLASS_COLOURS"]

logger = get_logger(__name__)

#: Target thumbnail width in pixels.
THUMBNAIL_WIDTH = 620

#: Must match the susceptibility class palette in the frontend.
CLASS_COLOURS = {
    "very_low": (79, 125, 99),
    "low": (143, 174, 93),
    "moderate": (217, 180, 74),
    "high": (217, 121, 58),
    "very_high": (194, 65, 45),
}


def _png(rgba: np.ndarray) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


class DistrictService:
    """Map thumbnails and point sampling over the district grid.

    Parameters
    ----------
    cfg : Config
        Resolved project configuration.
    prediction_service : PredictionService
        Supplies the threshold and landslide inventory once loaded.
    """

    def __init__(self, cfg, prediction_service) -> None:
        self.cfg = cfg
        self.prediction = prediction_service
        self.status = "idle"
        self.error: str | None = None
        self._images: dict[str, bytes] = {}
        self._grid: dict | None = None
        self._layers: dict[str, dict] = {}
        self._transformer = None

    def start(self) -> None:
        self.status = "loading"
        threading.Thread(target=self._build, name="district-thumbnails", daemon=True).start()

    # ------------------------------------------------------------------
    def _build(self) -> None:
        try:
            cfg = self.cfg
            slope_path = cfg.raster_path("slope")
            if not slope_path.exists():
                raise FileNotFoundError(f"slope raster not found: {slope_path}")

            with rasterio.open(slope_path) as src:
                factor = max(1, math.ceil(src.width / THUMBNAIL_WIDTH))
                shape = (math.ceil(src.height / factor), math.ceil(src.width / factor))
                slope = src.read(1, out_shape=shape, resampling=Resampling.nearest).astype("float64")
                inside = np.isfinite(slope)
                if src.nodata is not None:
                    inside &= ~np.isclose(slope, src.nodata)
                bounds = src.bounds
                crs = src.crs.to_string()

            self._grid = {
                "crs": crs,
                "bounds": [bounds.left, bounds.bottom, bounds.right, bounds.top],
                "image_size": [shape[1], shape[0]],
            }

            hillshade_path = cfg.processed_dir / "hillshade_uk.tif"
            if hillshade_path.exists():
                with rasterio.open(hillshade_path) as src:
                    shade = src.read(1, out_shape=shape, resampling=Resampling.average).astype("float64")
                    valid = inside & np.isfinite(shade)
                    if src.nodata is not None:
                        valid &= ~np.isclose(shade, src.nodata)
                source = "hillshade_uk.tif"
            else:
                # Fall back to slope as a relief proxy: steeper renders darker.
                shade = 255.0 - np.clip(slope, 0, 45) / 45.0 * 200.0
                valid = inside
                source = "slope_uk.tif (hillshade unavailable)"

            lo, hi = np.nanpercentile(shade[valid], [2, 98]) if valid.any() else (0.0, 255.0)
            grey = np.clip((shade - lo) / max(hi - lo, 1e-9), 0, 1)
            rgba = np.zeros((*shape, 4), dtype=np.uint8)
            rgba[..., 0] = (52 + grey * 150).astype(np.uint8)
            rgba[..., 1] = (56 + grey * 148).astype(np.uint8)
            rgba[..., 2] = (60 + grey * 140).astype(np.uint8)
            rgba[..., 3] = np.where(inside, 255, 0).astype(np.uint8)
            self._images["relief"] = _png(rgba)
            self._layers["relief"] = {"label": "Terrain relief", "source": source}

            self._build_susceptibility_layer(shape)
            self.status = "ready"
            logger.info("District thumbnails ready (%d x %d px)", shape[1], shape[0])
        except Exception as exc:  # noqa: BLE001 - surfaced through the API
            logger.exception("District map could not be built")
            self.status = "error"
            self.error = str(exc)

    def _build_susceptibility_layer(self, shape: tuple[int, int]) -> None:
        """Add the Stage 7 district map as an overlay, if it has been produced."""
        rasters_dir = self.cfg.rasters_dir
        path = rasters_dir / "susceptibility.tif"
        partial = False
        if not path.exists():
            candidates = sorted(rasters_dir.glob("susceptibility_partial_*.tif"))
            if not candidates:
                return
            path, partial = candidates[0], True

        with rasterio.open(path) as src:
            values = src.read(1, out_shape=shape, resampling=Resampling.average).astype("float64")
            ok = np.isfinite(values)
            if src.nodata is not None:
                ok &= ~np.isclose(values, src.nodata) & (values >= 0)

        # The threshold comes from the bundles; wait for them rather than guess.
        threshold = None
        for _ in range(600):
            state = self.prediction._state
            if state is not None:
                threshold = state["threshold"]
                break
            if self.prediction.status == "error":
                return
            threading.Event().wait(0.5)
        if threshold is None:
            return

        from src.inference.raster import susceptibility_classes

        rgba = np.zeros((*shape, 4), dtype=np.uint8)
        for key, (low, high) in susceptibility_classes(threshold).items():
            mask = ok & (values >= low) & (values < high)
            rgba[mask, :3] = CLASS_COLOURS[key]
            rgba[mask, 3] = 205
        self._images["susceptibility"] = _png(rgba)
        self._layers["susceptibility"] = {
            "label": "Model susceptibility (Stage 7 district map)",
            "source": path.name,
            "partial": partial,
        }

    # ------------------------------------------------------------------
    def meta(self) -> dict:
        points = self.prediction.landslide_points()
        return {
            "status": self.status,
            "error": self.error,
            "grid": self._grid,
            "layers": self._layers,
            "landslides": points.round(1).tolist() if points is not None else [],
        }

    def image(self, layer: str) -> bytes | None:
        return self._images.get(layer)

    def _lonlat(self, x: float, y: float) -> tuple[float, float] | None:
        try:
            if self._transformer is None:
                from pyproj import Transformer

                crs = self._grid["crs"] if self._grid else self.cfg.get("grid", "crs")
                self._transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
            lon, lat = self._transformer.transform(x, y)
            return float(lon), float(lat)
        except Exception:  # noqa: BLE001 - coordinates are a display nicety
            return None

    def sample(self, x: float, y: float) -> dict:
        """Read the 14 conditioning factors at a projected coordinate."""
        cfg = self.cfg
        left, bottom, right, top = cfg.get("grid", "bounds")
        if not (left <= x <= right and bottom <= y <= top):
            return {"inside": False, "x": x, "y": y, "factors": None, "missing": [], "issues": ["Outside the district grid."]}

        factors: dict = {}
        missing = []
        for factor in cfg.raster_factors:
            value = float(sample_raster_at_points(cfg.raster_path(factor), np.array([x]), np.array([y]))[0])
            if not np.isfinite(value):
                missing.append(factor)
                continue
            factors[factor] = int(round(value)) if factor in CATEGORICAL else value

        issues = []
        for factor, value in factors.items():
            bounds = FACTORS[factor].get("bounds")
            if bounds and not (bounds[0] <= value <= bounds[1]):
                issues.append(
                    f"{FACTORS[factor]['label']} reads {value:g} here, outside physical limits "
                    "(a raster artefact); this cell cannot be scored."
                )
        if missing:
            issues.append("No data for " + ", ".join(FACTORS[m]["label"] for m in missing) + " at this cell.")

        nearest = None
        within_1km = None
        points = self.prediction.landslide_points()
        if points is not None and len(points):
            distances = np.hypot(points[:, 0] - x, points[:, 1] - y)
            nearest = float(distances.min())
            within_1km = int((distances <= 1000).sum())

        lonlat = self._lonlat(x, y)
        return {
            "inside": not missing,
            "x": x,
            "y": y,
            "lon": lonlat[0] if lonlat else None,
            "lat": lonlat[1] if lonlat else None,
            "factors": factors if not missing else None,
            "missing": missing,
            "issues": issues,
            "nearest_landslide_m": nearest,
            "landslides_within_1km": within_1km,
        }
