"""Stage 7: windowed district-wide inference.

The prediction grid is 3,738 x 5,554 = 20,760,852 pixels, of which 10,929,202
fall inside the district. Materialising a 16-band float32 feature stack for all
of them is roughly 1.25 GB before any model runs — which is exactly what
blocked the previous attempt at producing a district map, where the pipeline
stopped at a missing ``feature_stack.tif``.

Nothing of the sort is built here. Inference walks the grid in horizontal
strips: each strip is read from the source rasters, encoded, predicted and
written out before the next is read, so peak memory is set by the strip height
rather than by the district. There is no intermediate stack on disk and none in
memory.

Two outputs are produced, and they are meant to be read together:

``susceptibility.tif``
    The calibrated aggregated prediction.
``confidence.tif``
    Derived from disagreement across the four calibrated thematic models. High
    susceptibility with high confidence means act; high susceptibility with low
    confidence means send a survey team before committing resources.

Nodata handling is per-source-file throughout. This project's layers use both
-9999 and -32768 depending on which toolchain produced them, and a pixel is
written as nodata if *any* contributing factor is nodata there — a prediction
made from a partially-missing feature vector would be silently wrong rather
than visibly absent.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

from src.features.themes import BENCHMARK_NAME
from src.models.bundle import load_bundle
from src.models.combiners import confidence_from_disagreement, disagreement
from src.utils.logging_setup import get_logger

__all__ = ["run_inference_stage", "predict_district"]

logger = get_logger(__name__)


def _open_sources(cfg, factors=None) -> dict[str, rasterio.DatasetReader]:
    """Open the named conditioning-factor rasters, keyed by factor name.

    Parameters
    ----------
    cfg : Config
        Resolved project configuration.
    factors : iterable of str, optional
        Which factors to open. Defaults to all configured factors.
    """
    names = list(factors) if factors is not None else cfg.raster_factors
    return {factor: rasterio.open(cfg.raster_path(factor)) for factor in names}


def _read_block(
    sources: dict, window: Window
) -> tuple[pd.DataFrame, np.ndarray, tuple[int, int]]:
    """Read one strip from every raster and build a raw-factor frame.

    Parameters
    ----------
    sources : dict
        Open rasterio datasets keyed by factor name.
    window : rasterio.windows.Window
        The strip to read.

    Returns
    -------
    frame : pandas.DataFrame
        One row per pixel in the strip, one column per conditioning factor.
    valid : numpy.ndarray
        Boolean mask, flattened, True where every factor has real data.
    shape : tuple of int
        ``(height, width)`` of the strip, for reshaping predictions back.
    """
    columns: dict[str, np.ndarray] = {}
    valid: np.ndarray | None = None

    for factor, src in sources.items():
        band = src.read(1, window=window).astype("float64")
        # Each raster's own declared nodata, read from its header. Never a
        # global constant: the hydrological layers use -32768 while the rest
        # use -9999, and mixing them up admits -32768 as a real measurement.
        ok = np.isfinite(band)
        if src.nodata is not None:
            ok &= ~np.isclose(band, src.nodata)
        valid = ok if valid is None else (valid & ok)
        columns[factor] = band.ravel()

    height, width = next(iter(columns.values())).size // window.width, window.width
    frame = pd.DataFrame(columns)
    return frame, valid.ravel(), (int(window.height), int(width))


def usable_themes(cfg) -> tuple[list[str], bool]:
    """Determine which thematic models can actually be run.

    A theme is usable only if every raster it needs is on disk. Returns the
    usable themes and whether the full-feature benchmark can also run.

    Parameters
    ----------
    cfg : Config
        Resolved project configuration.

    Returns
    -------
    themes : list of str
        Themes whose rasters are all present.
    full_available : bool
        Whether every raster is present, so the benchmark model can run.
    """
    missing = set(cfg.missing_rasters())
    usable = [
        theme
        for theme in cfg.theme_names
        if not (set(cfg.get("themes", theme, "factors")) & missing)
    ]
    return usable, not missing


def predict_district(
    cfg,
    block_height: int | None = None,
    themes: list[str] | None = None,
    use_full_model: bool = True,
    suffix: str = "",
) -> dict:
    """Run windowed inference across the district.

    Parameters
    ----------
    cfg : Config
        Resolved project configuration.
    block_height : int, optional
        Strip height in pixels. Defaults to ``inference.block_height``.
    themes : list of str, optional
        Thematic models to run. Defaults to all configured themes. Passing a
        subset produces a partial map, which callers must label as such.
    use_full_model : bool, default True
        Whether to include the full-feature benchmark in the aggregate. False
        when some rasters are unavailable, since the benchmark needs all 14.
    suffix : str, optional
        Appended to output filenames, so a partial run can never overwrite or
        be mistaken for the complete district map.

    Returns
    -------
    dict
        Summary statistics and output paths.
    """
    themes = list(themes) if themes is not None else list(cfg.theme_names)
    needed = themes + ([BENCHMARK_NAME] if use_full_model else [])
    bundles = {}
    for name in needed:
        path = cfg.models_dir / f"{name}.joblib"
        if not path.exists():
            raise FileNotFoundError(
                f"model bundle not found: {path}. Run stage 4 and stage 5 first."
            )
        bundles[name] = load_bundle(path)

    threshold = bundles[needed[0]].threshold
    if threshold is None:
        raise ValueError(
            "models carry no operating threshold; run stage 5 before inference"
        )

    block_height = int(
        block_height or cfg.get("inference", "block_height", default=256)
    )
    nodata_out = float(cfg.get("inference", "nodata", default=-9999.0))

    # Only open the rasters the selected models actually need. A partial run
    # must not fail merely because an unrelated layer is absent.
    required_factors = sorted(
        {f for theme in themes for f in cfg.get("themes", theme, "factors")}
    ) if not use_full_model else list(cfg.raster_factors)
    sources = _open_sources(cfg, required_factors)
    try:
        # Any open source defines the grid — they are all verified aligned in
        # Stage 1 — so a partial run does not depend on one specific layer.
        reference = sources.get("slope") or next(iter(sources.values()))
        profile = reference.profile.copy()
        height, width = reference.height, reference.width
        profile.update(
            dtype="float32",
            count=1,
            nodata=nodata_out,
            compress=cfg.get("inference", "compress", default="DEFLATE"),
            predictor=int(cfg.get("inference", "predictor", default=3)),
            tiled=bool(cfg.get("inference", "tiled", default=True)),
            blockxsize=256,
            blockysize=256,
        )

        cfg.rasters_dir.mkdir(parents=True, exist_ok=True)
        susceptibility_path = cfg.rasters_dir / f"susceptibility{suffix}.tif"
        disagreement_path = cfg.rasters_dir / f"disagreement{suffix}.tif"

        n_valid_total = 0
        susceptibility_sum = 0.0
        disagreement_values: list[np.ndarray] = []

        logger.info(
            "Inference over %d x %d px in %d strips of %d rows",
            width,
            height,
            int(np.ceil(height / block_height)),
            block_height,
        )

        with rasterio.open(susceptibility_path, "w", **profile) as dst_susc, \
             rasterio.open(disagreement_path, "w", **profile) as dst_disagree:

            for start in range(0, height, block_height):
                rows = min(block_height, height - start)
                window = Window(0, start, width, rows)
                frame, valid, shape = _read_block(sources, window)

                susc_block = np.full(frame.shape[0], nodata_out, dtype="float32")
                disagree_block = np.full(frame.shape[0], nodata_out, dtype="float32")

                if valid.any():
                    # The encoder's schema covers all 14 factors. On a partial
                    # run only some rasters are open, so the absent factors are
                    # supplied as NaN: they flow through encoding but belong to
                    # no model being run, so nothing consumes them. Reindexing
                    # here keeps the encoder's strict column check intact
                    # rather than weakening it for one code path.
                    subset = frame.loc[valid].reindex(
                        columns=list(cfg.raster_factors)
                    )
                    # Each bundle encodes internally, so a model can never be
                    # applied to features its own encoder did not produce.
                    thematic = np.column_stack(
                        [bundles[t].predict_proba(subset) for t in themes]
                    )
                    if use_full_model:
                        full = bundles[BENCHMARK_NAME].predict_proba(subset)
                        combined = 0.5 * (full + thematic.mean(axis=1))
                    else:
                        combined = thematic.mean(axis=1)
                    spread = disagreement(thematic, "std")

                    susc_block[valid] = combined.astype("float32")
                    disagree_block[valid] = spread.astype("float32")

                    n_valid_total += int(valid.sum())
                    susceptibility_sum += float(combined.sum())
                    # Subsample for the percentile used to scale confidence;
                    # holding every value would defeat the point of windowing.
                    disagreement_values.append(spread[:: max(1, spread.size // 5000)])

                dst_susc.write(susc_block.reshape(shape), 1, window=window)
                dst_disagree.write(disagree_block.reshape(shape), 1, window=window)

                if (start // block_height) % 5 == 0:
                    logger.info(
                        "  rows %5d-%5d  valid %7d  running mean susceptibility %.4f",
                        start,
                        start + rows,
                        n_valid_total,
                        susceptibility_sum / max(n_valid_total, 1),
                    )
    finally:
        for src in sources.values():
            src.close()

    # ------------------------------------------------------------------
    # second pass: confidence from the disagreement raster
    # ------------------------------------------------------------------
    # A separate pass rather than a second output in pass one, because the
    # scaling ceiling is not knowable until the whole district has been seen.
    sample = np.concatenate(disagreement_values) if disagreement_values else np.array([0.0])
    ceiling = float(np.nanpercentile(sample, 99))
    logger.info(
        "Disagreement ceiling for confidence scaling: %.4f (99th percentile)", ceiling
    )

    confidence_path = cfg.rasters_dir / f"confidence{suffix}.tif"
    with rasterio.open(disagreement_path) as src:
        profile = src.profile.copy()
        with rasterio.open(confidence_path, "w", **profile) as dst:
            for _, window in src.block_windows(1):
                block = src.read(1, window=window)
                out = np.full(block.shape, nodata_out, dtype="float32")
                ok = ~np.isclose(block, nodata_out) & np.isfinite(block)
                if ok.any():
                    out[ok] = confidence_from_disagreement(
                        block[ok], max_possible=ceiling
                    ).astype("float32")
                dst.write(out, 1, window=window)

    summary = {
        "themes_used": themes,
        "full_model_included": use_full_model,
        "is_partial": bool(suffix),
        "grid": {"width": width, "height": height},
        "valid_pixels": n_valid_total,
        "valid_fraction": round(n_valid_total / (width * height), 4),
        "mean_susceptibility": round(susceptibility_sum / max(n_valid_total, 1), 5),
        "threshold": threshold,
        "disagreement_ceiling_p99": round(ceiling, 5),
        "outputs": {
            "susceptibility": str(susceptibility_path),
            "disagreement": str(disagreement_path),
            "confidence": str(confidence_path),
        },
    }
    logger.info(
        "Inference complete: %d valid pixels, mean susceptibility %.4f",
        n_valid_total,
        summary["mean_susceptibility"],
    )
    return summary


def run_inference_stage(cfg, allow_partial: bool = True) -> dict:
    """Stage 7 entry point.

    Parameters
    ----------
    cfg : Config
        Resolved project configuration.
    allow_partial : bool, default True
        When some rasters are unavailable, run over whatever themes are
        complete and write clearly-suffixed partial outputs, rather than
        producing nothing. A partial map is useful — it still validates the
        windowed pipeline at full district scale and shows real spatial
        structure — but it is not the deliverable, so it never shares a
        filename with the complete map.

    Returns
    -------
    dict
        Report written to ``outputs/reports/inference.json``.

    Raises
    ------
    FileNotFoundError
        If rasters are missing and ``allow_partial`` is False, or if no theme
        is complete.
    """
    missing = cfg.missing_rasters()
    themes, full_available = usable_themes(cfg)
    suffix = ""

    if missing:
        if not allow_partial:
            raise FileNotFoundError(
                "district inference needs all 14 conditioning-factor rasters; "
                f"missing: {sorted(missing)}"
            )
        if not themes:
            raise FileNotFoundError(
                "no thematic model has all its rasters available; "
                f"missing: {sorted(missing)}"
            )
        suffix = f"_partial_{len(themes)}of{len(cfg.theme_names)}themes"
        logger.warning(
            "PARTIAL RUN: %d raster(s) unavailable (%s). Running %d of %d themes "
            "(%s) without the full-feature benchmark. Outputs carry the suffix "
            "%r and are NOT the district deliverable.",
            len(missing),
            sorted(missing),
            len(themes),
            len(cfg.theme_names),
            themes,
            suffix,
        )

    summary = predict_district(
        cfg,
        themes=themes,
        use_full_model=full_available,
        suffix=suffix,
    )
    summary["missing_rasters"] = sorted(missing)
    if missing:
        summary["partial_warning"] = (
            f"Partial map: built from {len(themes)} of {len(cfg.theme_names)} "
            f"themes ({themes}) because these rasters were unavailable: "
            f"{sorted(missing)}. The full-feature benchmark was excluded. "
            "Validates the pipeline at full district scale; not the deliverable."
        )

    # Area breakdown by susceptibility class, at the derived threshold and at
    # conventional quantile classes. Area is what a planner allocates against,
    # so a probability distribution alone is not a usable answer.
    with rasterio.open(summary["outputs"]["susceptibility"]) as src:
        data = src.read(1)
        nodata = src.nodata
        res_x, res_y = src.res
    ok = np.isfinite(data) & ~np.isclose(data, nodata)
    values = data[ok]
    cell_km2 = res_x * res_y / 1e6

    threshold = float(summary["threshold"])
    classes = {
        "very_low": (0.0, 0.05),
        "low": (0.05, 0.10),
        "moderate": (0.10, threshold),
        "high": (threshold, 0.40),
        "very_high": (0.40, 1.01),
    }
    breakdown = {}
    for name, (low, high) in classes.items():
        count = int(((values >= low) & (values < high)).sum())
        breakdown[name] = {
            "range": [round(low, 4), round(high, 4)],
            "pixels": count,
            "area_km2": round(count * cell_km2, 1),
            "share": round(count / max(values.size, 1), 4),
        }
    summary["susceptibility_classes"] = breakdown
    summary["flagged_area_km2"] = round(
        float((values >= threshold).sum()) * cell_km2, 1
    )
    summary["flagged_share"] = round(float((values >= threshold).mean()), 4)

    logger.info(
        "Above threshold %.4f: %.0f km2 (%.1f%% of the district)",
        threshold,
        summary["flagged_area_km2"],
        100 * summary["flagged_share"],
    )

    out_path = cfg.reports_dir / f"inference{suffix}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=str)
    logger.info("Inference report written: %s", out_path)

    return summary
