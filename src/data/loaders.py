"""Loading the delivered data assets.

The data is treated as read-only validated input. Nothing here regenerates a
conditioning factor; the job is to load, clean explicitly, and make every
removal auditable.

The one cleaning action taken at load time is documented in
``docs/methodology.md`` §2.8: four rows carry physically impossible negative
elevation (a DEM interpolation artefact over water) and are dropped. They are
dropped *here*, in the pipeline, with their ``sample_id`` values logged, rather
than by editing the delivered CSV — so the source asset stays untouched and the
removal appears in every run log.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import rasterio

from src.utils.logging_setup import get_logger

__all__ = [
    "CleaningReport",
    "RasterMeta",
    "load_training_data",
    "raster_metadata",
    "all_raster_metadata",
    "sample_raster_at_points",
    "sample_factors_at_points",
]

logger = get_logger(__name__)


@dataclass
class CleaningReport:
    """Record of what ``load_training_data`` removed and why.

    Attributes
    ----------
    rows_in : int
        Row count in the delivered CSV.
    rows_out : int
        Row count after cleaning.
    dropped : dict of str to list of int
        Reason mapped to the ``sample_id`` values removed for that reason.
    positives_in, positives_out : int
        Positive-class counts before and after, tracked separately because
        losing positives matters far more than losing negatives at 1:10.8.
    """

    rows_in: int = 0
    rows_out: int = 0
    positives_in: int = 0
    positives_out: int = 0
    dropped: dict[str, list[int]] = field(default_factory=dict)

    def as_dict(self) -> dict:
        """Return a JSON-serialisable summary."""
        return {
            "rows_in": self.rows_in,
            "rows_out": self.rows_out,
            "positives_in": self.positives_in,
            "positives_out": self.positives_out,
            "negatives_in": self.rows_in - self.positives_in,
            "negatives_out": self.rows_out - self.positives_out,
            "dropped": {k: list(map(int, v)) for k, v in self.dropped.items()},
            "dropped_total": sum(len(v) for v in self.dropped.values()),
        }


def load_training_data(
    cfg, drop_invalid: bool = True
) -> tuple[pd.DataFrame, CleaningReport]:
    """Load the labelled training dataset.

    Parameters
    ----------
    cfg : Config
        Resolved project configuration.
    drop_invalid : bool, default True
        Whether to remove rows failing physical-plausibility checks. Set False
        only to inspect the raw delivered file.

    Returns
    -------
    df : pandas.DataFrame
        The cleaned dataset, index reset.
    report : CleaningReport
        What was removed and why.

    Notes
    -----
    Negative elevation is the only plausibility rule applied. It is safe
    because elevation is measured above sea level and the district's minimum
    real elevation is zero; the four affected rows sit on permanent water at
    the estuaries. No other row is removed on statistical grounds — outliers
    in slope, SPI or curvature are real terrain, not errors.
    """
    path = cfg.training_csv
    logger.info("Loading training data: %s", path)
    df = pd.read_csv(path)

    report = CleaningReport(
        rows_in=len(df),
        positives_in=int(df["label"].sum()),
    )
    logger.info(
        "Delivered file: %d rows (%d positive / %d negative, 1:%.2f)",
        report.rows_in,
        report.positives_in,
        report.rows_in - report.positives_in,
        (report.rows_in - report.positives_in) / max(report.positives_in, 1),
    )

    if drop_invalid:
        # Physically impossible: elevation is metres above sea level.
        invalid = df["elevation"] < 0
        if invalid.any():
            ids = df.loc[invalid, "sample_id"].tolist()
            labels = df.loc[invalid, "label"].tolist()
            report.dropped["negative_elevation"] = ids
            logger.warning(
                "Dropping %d row(s) with negative elevation (DEM artefact over "
                "water): sample_id=%s, labels=%s",
                len(ids),
                ids,
                labels,
            )
            df = df.loc[~invalid].copy()

    df = df.reset_index(drop=True)
    report.rows_out = len(df)
    report.positives_out = int(df["label"].sum())
    logger.info(
        "Working dataset: %d rows (%d positive / %d negative, 1:%.2f)",
        report.rows_out,
        report.positives_out,
        report.rows_out - report.positives_out,
        (report.rows_out - report.positives_out) / max(report.positives_out, 1),
    )
    return df, report


@dataclass(frozen=True)
class RasterMeta:
    """Grid and type metadata for one raster.

    Captured so that alignment can be asserted rather than assumed. Misaligned
    rasters are the most destructive silent failure in raster ML: inference
    completes, the map is written, and every pixel is sampling the wrong place.
    """

    name: str
    path: Path
    width: int
    height: int
    crs: str
    transform: tuple[float, ...]
    resolution: tuple[float, float]
    bounds: tuple[float, float, float, float]
    dtype: str
    nodata: float | None
    count: int

    def grid_key(self) -> tuple:
        """Return the tuple that must match across all aligned rasters.

        Transform values are rounded to 6 decimal places (micrometre precision
        in a metre-based CRS) to absorb float round-trip noise from GeoTIFF
        storage without admitting any real offset.
        """
        return (
            self.width,
            self.height,
            self.crs,
            tuple(round(v, 6) for v in self.transform),
        )

    def as_dict(self) -> dict:
        """Return a JSON-serialisable summary."""
        return {
            "name": self.name,
            "file": self.path.name,
            "width": self.width,
            "height": self.height,
            "crs": self.crs,
            "resolution_m": list(self.resolution),
            "bounds": list(self.bounds),
            "dtype": self.dtype,
            "nodata": None if self.nodata is None else float(self.nodata),
            "bands": self.count,
        }


def raster_metadata(name: str, path: Path) -> RasterMeta:
    """Read one raster's grid metadata without loading its pixels.

    Parameters
    ----------
    name : str
        Conditioning-factor name, used for reporting.
    path : Path
        Path to the GeoTIFF.

    Returns
    -------
    RasterMeta
        The raster's grid description.
    """
    with rasterio.open(path) as src:
        return RasterMeta(
            name=name,
            path=Path(path),
            width=src.width,
            height=src.height,
            crs=str(src.crs),
            transform=tuple(src.transform)[:6],
            resolution=(float(src.res[0]), float(src.res[1])),
            bounds=tuple(float(v) for v in src.bounds),
            dtype=str(src.dtypes[0]),
            nodata=None if src.nodata is None else float(src.nodata),
            count=src.count,
        )


def all_raster_metadata(cfg, factors: Iterable[str] | None = None) -> dict[str, RasterMeta]:
    """Read grid metadata for every configured conditioning factor.

    Parameters
    ----------
    cfg : Config
        Resolved project configuration.
    factors : iterable of str, optional
        Subset of factors to read. Defaults to all configured factors.

    Returns
    -------
    dict
        Factor name mapped to its :class:`RasterMeta`.
    """
    names = list(factors) if factors is not None else cfg.raster_factors
    return {name: raster_metadata(name, cfg.raster_path(name)) for name in names}


def sample_raster_at_points(
    path: Path,
    xs: np.ndarray,
    ys: np.ndarray,
    nodata_to_nan: bool = True,
) -> np.ndarray:
    """Sample a raster at projected point coordinates.

    Parameters
    ----------
    path : Path
        Path to the GeoTIFF.
    xs, ys : numpy.ndarray
        Easting and northing in the raster's own CRS (EPSG:32643 here).
    nodata_to_nan : bool, default True
        Convert the raster's declared nodata to NaN. The nodata value is read
        from the file header, never assumed — this project's layers use both
        -9999 and -32768 depending on which toolchain produced them.

    Returns
    -------
    numpy.ndarray
        Sampled values as float64, with nodata as NaN where requested.
    """
    with rasterio.open(path) as src:
        nodata = src.nodata
        # rasterio.sample takes (x, y) pairs and returns one array per point.
        coords = np.column_stack([xs, ys])
        values = np.fromiter(
            (rec[0] for rec in src.sample(coords)),
            dtype="float64",
            count=len(coords),
        )
    if nodata_to_nan and nodata is not None:
        values = np.where(np.isclose(values, nodata), np.nan, values)
    return values


def sample_factors_at_points(
    cfg,
    df: pd.DataFrame,
    factors: Iterable[str],
    x_col: str = "X",
    y_col: str = "Y",
) -> pd.DataFrame:
    """Sample several conditioning-factor rasters at the sample coordinates.

    Used to verify that the delivered training CSV was actually built from the
    rasters the configuration points at. If the CSV was sampled from one
    rainfall layer and inference runs on another, training and prediction sit
    on different distributions — a failure that produces no error and no
    visible artefact, only a quietly wrong map.

    Parameters
    ----------
    cfg : Config
        Resolved project configuration.
    df : pandas.DataFrame
        Table containing the coordinate columns.
    factors : iterable of str
        Conditioning factors to sample.
    x_col, y_col : str
        Coordinate column names.

    Returns
    -------
    pandas.DataFrame
        One column per factor, indexed like ``df``.
    """
    xs = df[x_col].to_numpy()
    ys = df[y_col].to_numpy()
    out = {}
    for factor in factors:
        path = cfg.raster_path(factor)
        logger.debug("Sampling %s at %d points", path.name, len(xs))
        out[factor] = sample_raster_at_points(path, xs, ys)
    return pd.DataFrame(out, index=df.index)
