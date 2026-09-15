"""Encoding conditioning factors into model features.

The encoder is a fitted object, not a function, because it must travel with the
model. A model that can be loaded without its encoder will eventually be used
without one, and the resulting predictions look entirely plausible while being
wrong — a particularly dangerous failure for raster output, where the map still
looks like a map.

Four encodings are applied. Three are non-obvious and are justified at length
in ``docs/methodology.md`` §2.2, §2.4 and §3.3:

1. **Circular aspect.** Aspect is degrees on a circle; as a scalar, 359 and 1
   are 358 apart despite being adjacent. Encoded as ``sin``/``cos``.
2. **Flat-cell handling.** Cells with zero slope have no defined aspect and
   carry a literal 270.0 placeholder from the terrain software. They are
   flagged with ``is_flat`` and their aspect components zeroed, so the
   placeholder cannot be read as a real westward bearing.
3. **log1p on SPI.** Four orders of magnitude, skew 46.3 to 1.29. Trees are
   invariant to monotonic transforms so this does not move tree performance;
   it is applied so SHAP values and partial dependence read on a physically
   meaningful scale, and so any scale-sensitive component sees a sane range.
4. **Categoricals stay categorical.** ``lithology`` and ``LULC`` are nominal
   codes. They are converted to string labels for native categorical handling,
   never to an integer a model would read as an ordering.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.utils.logging_setup import get_logger

__all__ = ["FeatureEncoder", "build_encoder"]

logger = get_logger(__name__)

#: ESA WorldCover class code -> readable label. Used so that categorical levels
#: are self-describing in SHAP plots and model dumps rather than bare integers.
LULC_LABELS = {
    10: "tree_cover",
    20: "shrubland",
    30: "grassland",
    40: "cropland",
    50: "built_up",
    60: "bare_sparse",
    70: "snow_ice",
    80: "permanent_water",
    90: "herbaceous_wetland",
    95: "mangroves",
    100: "moss_lichen",
}

#: GLiM lithology class code -> readable label, from processed/lithology_classes.json.
LITHOLOGY_LABELS = {
    1: "mt_metamorphic",
    2: "pa_acid_plutonic",
    3: "su_siliciclastic_sedimentary",
    4: "vb_basic_volcanic",
    5: "wb_water_bodies",
}


class FeatureEncoder(BaseEstimator, TransformerMixin):
    """Turn raw conditioning factors into model features.

    Parameters
    ----------
    circular_columns : list of str
        Columns measured in degrees on a circle, encoded to sin/cos pairs.
    log1p_columns : list of str
        Non-negative, heavily skewed columns to log-transform.
    categorical_columns : list of str
        Nominal code columns converted to string labels.
    flat_slope_tolerance : float, default 0.0
        Slope at or below which a cell is treated as flat and its aspect
        components zeroed.
    lithology_unmapped_code : float, default -1.0
        Sentinel marking absent lithology coverage.
    lithology_unmapped_label : str
        Name given to that sentinel as a category. Named rather than numeric so
        that it is impossible to read a SHAP plot without noticing it is a
        coverage gap, not a rock type.
    drop_columns : list of str
        Columns that must never reach a model.

    Attributes
    ----------
    feature_names_ : list of str
        Output column names, fixed at fit time and enforced at transform time.
    categorical_features_ : list of str
        Which output columns are categorical, for learners that consume them
        natively.
    categories_ : dict
        Categorical column mapped to the levels seen during fit.
    """

    def __init__(
        self,
        circular_columns: list[str] | None = None,
        log1p_columns: list[str] | None = None,
        categorical_columns: list[str] | None = None,
        flat_slope_tolerance: float = 0.0,
        lithology_unmapped_code: float = -1.0,
        lithology_unmapped_label: str = "unmapped_coastal_gap",
        drop_columns: list[str] | None = None,
    ) -> None:
        self.circular_columns = circular_columns or ["aspect"]
        self.log1p_columns = log1p_columns or ["SPI"]
        self.categorical_columns = categorical_columns or ["lithology", "LULC"]
        self.flat_slope_tolerance = flat_slope_tolerance
        self.lithology_unmapped_code = lithology_unmapped_code
        self.lithology_unmapped_label = lithology_unmapped_label
        self.drop_columns = drop_columns or []

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _label_categorical(self, column: str, values: pd.Series) -> pd.Series:
        """Map a nominal code column to readable string labels.

        Unknown codes become ``"code_<n>"`` rather than raising, so that a
        district raster containing a class absent from the training sample
        still produces a defined category at inference instead of crashing
        mid-map.
        """
        if column == "lithology":
            lookup = dict(LITHOLOGY_LABELS)
            lookup[int(self.lithology_unmapped_code)] = self.lithology_unmapped_label
        elif column == "LULC":
            lookup = LULC_LABELS
        else:
            lookup = {}

        def _map(value: float) -> str:
            if not np.isfinite(value):
                return "missing"
            code = int(round(value))
            return lookup.get(code, f"code_{code}")

        return values.map(_map).astype("string")

    def _encode(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply every encoding rule. Shared by fit and transform."""
        out = pd.DataFrame(index=X.index)

        # --- pass through untouched numeric factors ----------------------
        handled = (
            set(self.circular_columns)
            | set(self.log1p_columns)
            | set(self.categorical_columns)
            | set(self.drop_columns)
            | {"label"}
        )
        for column in X.columns:
            if column in handled:
                continue
            out[column] = X[column].astype("float64")

        # --- circular encoding -------------------------------------------
        # A flat cell has no aspect. The terrain software emits a literal
        # 270.0 there; sin/cos of that would read as a genuine due-west
        # bearing, so the components are zeroed and the flatness is carried
        # explicitly by is_flat instead.
        for column in self.circular_columns:
            if column not in X.columns:
                continue
            radians = np.deg2rad(X[column].astype("float64").to_numpy())
            sin_values = np.sin(radians)
            cos_values = np.cos(radians)

            if "slope" in X.columns:
                is_flat = (
                    X["slope"].astype("float64").to_numpy()
                    <= self.flat_slope_tolerance
                ).astype("int8")
                sin_values = np.where(is_flat == 1, 0.0, sin_values)
                cos_values = np.where(is_flat == 1, 0.0, cos_values)
                out["is_flat"] = is_flat

            out[f"{column}_sin"] = sin_values
            out[f"{column}_cos"] = cos_values

        # --- log transform ------------------------------------------------
        for column in self.log1p_columns:
            if column not in X.columns:
                continue
            values = X[column].astype("float64").to_numpy()
            # Clip at zero rather than failing: a tiny negative from floating
            # point in an index defined as non-negative is noise, not data.
            out[f"{column}_log"] = np.log1p(np.clip(values, 0.0, None))

        # --- categoricals ---------------------------------------------------
        for column in self.categorical_columns:
            if column not in X.columns:
                continue
            out[column] = self._label_categorical(column, X[column].astype("float64"))

        return out

    # ------------------------------------------------------------------
    # sklearn API
    # ------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y=None) -> "FeatureEncoder":
        """Learn the output schema and categorical levels.

        Parameters
        ----------
        X : pandas.DataFrame
            Raw conditioning factors.
        y : ignored

        Returns
        -------
        FeatureEncoder
            The fitted encoder.
        """
        encoded = self._encode(X)
        self.feature_names_ = list(encoded.columns)
        self.categorical_features_ = [
            c for c in self.categorical_columns if c in encoded.columns
        ]
        self.categories_ = {
            c: sorted(encoded[c].dropna().unique().tolist())
            for c in self.categorical_features_
        }
        logger.info(
            "FeatureEncoder fitted: %d features (%d categorical) from %d raw columns",
            len(self.feature_names_),
            len(self.categorical_features_),
            X.shape[1],
        )
        for column, levels in self.categories_.items():
            logger.info("  %s levels (%d): %s", column, len(levels), levels)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Encode raw factors into model features.

        Parameters
        ----------
        X : pandas.DataFrame
            Raw conditioning factors.

        Returns
        -------
        pandas.DataFrame
            Encoded features in the exact column order fixed at fit time.

        Raises
        ------
        ValueError
            If the encoded frame lacks a column seen at fit time. Column order
            and membership are enforced rather than assumed, because a silently
            reordered feature matrix is the classic way a correct model starts
            producing a wrong map.
        """
        if not hasattr(self, "feature_names_"):
            raise ValueError("FeatureEncoder must be fitted before transform")

        encoded = self._encode(X)
        missing = [c for c in self.feature_names_ if c not in encoded.columns]
        if missing:
            raise ValueError(f"encoded frame is missing expected columns: {missing}")
        return encoded[self.feature_names_]

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Return output feature names, for sklearn interoperability."""
        return np.asarray(self.feature_names_, dtype=object)


def build_encoder(cfg) -> FeatureEncoder:
    """Construct a :class:`FeatureEncoder` from the project configuration.

    Parameters
    ----------
    cfg : Config
        Resolved project configuration.

    Returns
    -------
    FeatureEncoder
        An unfitted encoder wired to the configured encoding rules.
    """
    features = cfg.get("features", default={})
    return FeatureEncoder(
        circular_columns=list(features.get("circular", ["aspect"])),
        log1p_columns=list(features.get("log1p_transform", ["SPI"])),
        categorical_columns=list(features.get("categorical", ["lithology", "LULC"])),
        flat_slope_tolerance=float(features.get("flat_slope_tolerance", 0.0)),
        lithology_unmapped_code=float(features.get("lithology_unmapped_code", -1)),
        lithology_unmapped_label=str(
            features.get("lithology_unmapped_label", "unmapped_coastal_gap")
        ),
        drop_columns=list(features.get("forbidden_columns", [])),
    )
