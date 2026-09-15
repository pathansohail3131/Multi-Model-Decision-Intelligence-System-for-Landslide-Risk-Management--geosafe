"""Figures and cartographic output.

Two conventions apply throughout, both chosen so the figures cannot mislead:

* **Susceptibility and confidence are always produced as a pair.** A
  susceptibility map published alone invites exactly the over-confident reading
  this project exists to prevent.
* **Nodata is transparent, never a colour.** Rendering "no data" as a pale fill
  makes it indistinguishable from "low susceptibility", which is the difference
  between *we checked and it is safe* and *we never looked*.

Colour follows the job, not taste. Susceptibility and confidence are continuous
magnitude, so each gets a **single-hue sequential ramp, light to dark** — never
a rainbow, whose perceptual non-monotonicity invents boundaries that are not in
the data. Comparison charts use a fixed categorical order from a
CVD-validated palette, with direct labels so identity never rests on colour
alone.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import LinearSegmentedColormap

from src.utils.logging_setup import get_logger

__all__ = ["run_figures", "plot_raster"]

logger = get_logger(__name__)

# --- palette ---------------------------------------------------------------
# Sequential blue ramp (100 -> 700), light to dark, for magnitude.
BLUE_RAMP = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
    "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
    "#184f95", "#104281", "#0d366b",
]
# Second sequential context takes the next categorical hue, as its own ramp.
ORANGE_RAMP = [
    "#fde3d6", "#fbc9b0", "#f8ae8a", "#f59366", "#f17a45",
    "#eb6834", "#d95926", "#c04b1d", "#a53f17", "#8a3412",
]
# Categorical slots, validated: lightness band, chroma floor, CVD separation
# and normal-vision floor all PASS on the light surface.
SERIES = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a"}

INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#8a8a85"}
SURFACE = "#fcfcfb"

SUSCEPTIBILITY_CMAP = LinearSegmentedColormap.from_list("susc", BLUE_RAMP)
CONFIDENCE_CMAP = LinearSegmentedColormap.from_list("conf", ORANGE_RAMP)


def _style_axes(ax) -> None:
    """Make grid and spines recessive so the data carries the figure."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK["muted"])
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK["secondary"], labelsize=9, length=3, width=0.8)
    ax.grid(True, color="#e6e5e2", linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)


def plot_raster(
    path: Path,
    out_path: Path,
    title: str,
    subtitle: str = "",
    cmap=SUSCEPTIBILITY_CMAP,
    vmin: float | None = None,
    vmax: float | None = None,
    label: str = "",
    decimate: int = 4,
) -> Path:
    """Render one raster as a PNG map with nodata transparent.

    Parameters
    ----------
    path : Path
        Source GeoTIFF.
    out_path : Path
        Destination PNG.
    title, subtitle : str
        Figure title and a smaller line beneath it, used to carry the
        provisional/partial status where that applies.
    cmap : matplotlib colormap
        Single-hue sequential ramp.
    vmin, vmax : float, optional
        Colour limits. Defaults to the 1st and 99th percentiles, which stops a
        handful of extreme pixels from flattening the whole district into the
        bottom of the ramp.
    label : str
        Colourbar label.
    decimate : int, default 4
        Read-time downsampling factor. The grid is 3,738 x 5,554; rendering it
        at full resolution produces a figure no screen shows more detail on.

    Returns
    -------
    Path
        The written PNG.
    """
    with rasterio.open(path) as src:
        height = src.height // decimate
        width = src.width // decimate
        data = src.read(1, out_shape=(1, height, width)).astype("float64")
        nodata = src.nodata
        bounds = src.bounds

    mask = np.isfinite(data)
    if nodata is not None:
        mask &= ~np.isclose(data, nodata)
    # Masked array so nodata renders as the figure background, not as a value.
    shown = np.ma.masked_where(~mask, data)

    if vmin is None:
        vmin = float(np.nanpercentile(data[mask], 1)) if mask.any() else 0.0
    if vmax is None:
        vmax = float(np.nanpercentile(data[mask], 99)) if mask.any() else 1.0

    fig, ax = plt.subplots(figsize=(7.2, 9.6), dpi=140)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    image = ax.imshow(
        shown,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=(bounds.left, bounds.right, bounds.bottom, bounds.top),
        interpolation="nearest",
    )
    # Title sits above the subtitle with enough pad for both plus the axis
    # offset text matplotlib places at the top-left on large coordinate values.
    ax.set_title(title, fontsize=13, color=INK["primary"], pad=42, loc="left")
    if subtitle:
        ax.text(
            0.0, 1.028, subtitle, transform=ax.transAxes,
            fontsize=9, color=INK["secondary"], ha="left", va="bottom",
        )
    ax.set_xlabel("Easting (m, EPSG:32643)", fontsize=9, color=INK["secondary"])
    ax.set_ylabel("Northing (m, EPSG:32643)", fontsize=9, color=INK["secondary"])
    ax.tick_params(colors=INK["secondary"], labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK["muted"])

    bar = fig.colorbar(image, ax=ax, fraction=0.040, pad=0.03)
    bar.set_label(label, fontsize=9, color=INK["secondary"])
    bar.ax.tick_params(colors=INK["secondary"], labelsize=8)
    bar.outline.set_visible(False)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure written: %s", out_path)
    return out_path


def _plot_spatial_vs_random(cfg, out_path: Path) -> Path | None:
    """Plot the block-size sweep against the random-CV baseline.

    This is the project's headline diagnostic: it shows the optimism a random
    split injects, and how that optimism grows as blocks get large enough to
    genuinely separate training from test terrain.
    """
    path = cfg.reports_dir / "spatial_cv.json"
    if not path.exists():
        return None
    report = json.loads(path.read_text())

    sizes, auc_pr, roc = [], [], []
    for key, entry in report["spatial_cv_sweep"].items():
        sizes.append(int(key.replace("m", "")) / 1000.0)
        auc_pr.append(entry["summary"]["average_precision_mean"])
        roc.append(entry["summary"]["roc_auc_mean"])
    random_pr = report["random_cv"]["summary"]["average_precision_mean"]
    random_roc = report["random_cv"]["summary"]["roc_auc_mean"]

    fig, ax = plt.subplots(figsize=(8.0, 4.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    _style_axes(ax)

    ax.axhline(random_pr, color=SERIES["orange"], linewidth=2, linestyle="--")
    ax.axhline(random_roc, color=SERIES["blue"], linewidth=2, linestyle="--")
    ax.plot(sizes, auc_pr, color=SERIES["orange"], linewidth=2, marker="o", markersize=8)
    ax.plot(sizes, roc, color=SERIES["blue"], linewidth=2, marker="o", markersize=8)

    # Direct labels: identity never rests on colour alone.
    ax.text(sizes[-1], auc_pr[-1] + 0.012, "AUC-PR (spatial)", color=SERIES["orange"],
            fontsize=9, ha="right", va="bottom", fontweight="bold")
    ax.text(sizes[-1], roc[-1] + 0.012, "ROC-AUC (spatial)", color=SERIES["blue"],
            fontsize=9, ha="right", va="bottom", fontweight="bold")
    ax.text(sizes[0], random_pr + 0.008, f"random CV AUC-PR {random_pr:.3f}",
            color=SERIES["orange"], fontsize=8.5, va="bottom")
    ax.text(sizes[0], random_roc + 0.008, f"random CV ROC-AUC {random_roc:.3f}",
            color=SERIES["blue"], fontsize=8.5, va="bottom")

    ax.set_xscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels([f"{s:g}" for s in sizes])
    ax.set_xlabel("Spatial CV block size (km)", fontsize=10, color=INK["secondary"])
    ax.set_ylabel("Score", fontsize=10, color=INK["secondary"])
    ax.set_title(
        "Random cross-validation overstates performance",
        fontsize=13, color=INK["primary"], loc="left", pad=16,
    )
    ax.text(
        0.0, 1.02,
        "Dashed = random split. Solid = spatial block CV. The gap is leakage from clustered landslides.",
        transform=ax.transAxes, fontsize=9, color=INK["secondary"], va="bottom",
    )
    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure written: %s", out_path)
    return out_path


def _plot_ablation(cfg, out_path: Path) -> Path | None:
    """Horizontal bars of every combination rule, chosen one highlighted.

    Magnitude of one measure across named candidates, so this is a single-hue
    comparison, not a categorical identity problem. The bar that produces a
    confidence layer is distinguished by hue *and* by an explicit annotation.
    """
    path = cfg.reports_dir / "models_ablation.json"
    if not path.exists():
        return None
    report = json.loads(path.read_text())
    ablation = report["ablation"]

    names = sorted(ablation, key=lambda k: ablation[k]["average_precision"])
    scores = [ablation[n]["average_precision"] for n in names]
    chosen = report["best_producing_confidence"]

    colors = [
        SERIES["orange"] if n == chosen
        else (INK["muted"] if not ablation[n]["produces_confidence"] else SERIES["blue"])
        for n in names
    ]

    fig, ax = plt.subplots(figsize=(8.6, 4.8), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    _style_axes(ax)
    ax.grid(axis="y", visible=False)

    positions = np.arange(len(names))
    # 2px surface gap between adjacent bars, thin marks.
    ax.barh(positions, scores, height=0.68, color=colors, edgecolor=SURFACE, linewidth=2)
    for pos, score in zip(positions, scores):
        ax.text(score + 0.006, pos, f"{score:.4f}", va="center",
                fontsize=9, color=INK["primary"])

    ax.set_yticks(positions)
    ax.set_yticklabels([n.replace("_", " ") for n in names], fontsize=9.5,
                       color=INK["secondary"])
    ax.set_xlim(0, max(scores) * 1.16)
    ax.set_xlabel("AUC-PR (spatial block CV)", fontsize=10, color=INK["secondary"])
    ax.set_title("Combination-rule ablation", fontsize=13, color=INK["primary"],
                 loc="left", pad=16)
    ax.text(
        0.0, 1.02,
        f"Orange = adopted ({chosen.replace('_',' ')}). Grey = cannot produce a confidence layer.",
        transform=ax.transAxes, fontsize=9, color=INK["secondary"], va="bottom",
    )
    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure written: %s", out_path)
    return out_path


def _plot_model_strength(cfg, out_path: Path) -> Path | None:
    """Per-model AUC-PR with fold-to-fold spread shown as error bars.

    The spread is not decoration. With 417 positives over 5 spatial folds, a
    fold holds roughly 80 positives, so differences smaller than the error bars
    are not differences.
    """
    path = cfg.reports_dir / "models_ablation.json"
    if not path.exists():
        return None
    report = json.loads(path.read_text())
    models = report["individual_models"]

    names = sorted(models, key=lambda k: models[k]["summary"]["average_precision_mean"])
    means = [models[n]["summary"]["average_precision_mean"] for n in names]
    stds = [models[n]["summary"]["average_precision_std"] for n in names]
    counts = [models[n]["n_features"] for n in names]

    fig, ax = plt.subplots(figsize=(8.2, 4.4), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    _style_axes(ax)
    ax.grid(axis="y", visible=False)

    positions = np.arange(len(names))
    colors = [INK["muted"] if n == "full" else SERIES["blue"] for n in names]
    ax.barh(positions, means, height=0.66, color=colors, edgecolor=SURFACE, linewidth=2)
    ax.errorbar(means, positions, xerr=stds, fmt="none",
                ecolor=INK["secondary"], elinewidth=1.4, capsize=4)
    for pos, mean, std, n_feat in zip(positions, means, stds, counts):
        ax.text(mean + std + 0.02, pos, f"{mean:.3f}  ({n_feat}f)",
                va="center", fontsize=9, color=INK["primary"])

    ax.set_yticks(positions)
    ax.set_yticklabels(names, fontsize=10, color=INK["secondary"])
    ax.set_xlim(0, max(m + s for m, s in zip(means, stds)) * 1.25)
    ax.set_xlabel("AUC-PR (spatial block CV, mean +- SD across folds)",
                  fontsize=10, color=INK["secondary"])
    ax.set_title("Thematic model strength", fontsize=13, color=INK["primary"],
                 loc="left", pad=16)
    ax.text(0.0, 1.02, "Grey = full-feature benchmark. Bars overlap: fold spread is large.",
            transform=ax.transAxes, fontsize=9, color=INK["secondary"], va="bottom")
    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure written: %s", out_path)
    return out_path


def run_figures(cfg) -> dict:
    """Generate every figure from the artefacts on disk.

    Parameters
    ----------
    cfg : Config
        Resolved project configuration.

    Returns
    -------
    dict
        Figure name mapped to output path.
    """
    out: dict[str, str] = {}
    figures_dir = cfg.figures_dir
    figures_dir.mkdir(parents=True, exist_ok=True)

    rasters = {
        "susceptibility": (SUSCEPTIBILITY_CMAP, "Susceptibility (calibrated probability)"),
        "confidence": (CONFIDENCE_CMAP, "Confidence (1 = themes agree)"),
        "priority": (SUSCEPTIBILITY_CMAP, "Priority (susceptibility x confidence x exposure)"),
    }
    for name, (cmap, label) in rasters.items():
        candidates = sorted(cfg.rasters_dir.glob(f"{name}*.tif"))
        if not candidates:
            continue
        source = candidates[0]
        partial = "partial" in source.stem
        subtitle = (
            "PROVISIONAL — built from 2 of 4 themes; 2 rasters were unavailable"
            if partial
            else "Uttara Kannada district, 30.46 m grid, EPSG:32643"
        )
        out[name] = str(
            plot_raster(
                source,
                figures_dir / f"{name}_map.png",
                title=f"{name.capitalize()} — Uttara Kannada",
                subtitle=subtitle,
                cmap=cmap,
                label=label,
            )
        )

    for name, fn in (
        ("spatial_vs_random", _plot_spatial_vs_random),
        ("ablation", _plot_ablation),
        ("model_strength", _plot_model_strength),
    ):
        path = fn(cfg, figures_dir / f"{name}.png")
        if path is not None:
            out[name] = str(path)

    return out
