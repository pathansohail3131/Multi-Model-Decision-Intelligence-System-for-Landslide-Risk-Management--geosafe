"""Request validation for the prediction API.

Numeric factors are checked against physical limits only. Anything physically
possible is accepted and scored; values outside the training data are flagged
in the response instead of being rejected, so the caller can see that the model
is extrapolating.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from api.app.schemas.factors import CATEGORY_LEVELS, FACTORS

__all__ = ["FactorValues", "PredictRequest", "SensitivityRequest", "FactorName"]

FactorName = Literal[
    "LULC",
    "NDVI",
    "SPI",
    "TWI",
    "aspect",
    "dist_drainage",
    "dist_fault",
    "dist_road",
    "elevation",
    "lithology",
    "plan_curv",
    "profile_curv",
    "rainfall",
    "slope",
]


def _bounded(name: str):
    low, high = FACTORS[name]["bounds"]
    return Field(..., ge=low, le=high, description=FACTORS[name]["description"])


class FactorValues(BaseModel):
    """The 14 raw conditioning factors for one 30 m cell."""

    model_config = ConfigDict(extra="forbid")

    elevation: float = _bounded("elevation")
    slope: float = _bounded("slope")
    aspect: float = _bounded("aspect")
    plan_curv: float = _bounded("plan_curv")
    profile_curv: float = _bounded("profile_curv")
    TWI: float = _bounded("TWI")
    SPI: float = _bounded("SPI")
    dist_drainage: float = _bounded("dist_drainage")
    rainfall: float = _bounded("rainfall")
    lithology: int = Field(..., description=FACTORS["lithology"]["description"])
    dist_fault: float = _bounded("dist_fault")
    dist_road: float = _bounded("dist_road")
    LULC: int = Field(..., description=FACTORS["LULC"]["description"])
    NDVI: float = _bounded("NDVI")

    @field_validator("*", mode="before")
    @classmethod
    def _finite(cls, value):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("must be a finite number")
        return value

    @field_validator("lithology")
    @classmethod
    def _lithology_code(cls, value: int) -> int:
        if value not in CATEGORY_LEVELS["lithology"]:
            raise ValueError(f"unknown lithology code {value}; expected one of {sorted(CATEGORY_LEVELS['lithology'])}")
        return value

    @field_validator("LULC")
    @classmethod
    def _lulc_code(cls, value: int) -> int:
        if value not in CATEGORY_LEVELS["LULC"]:
            raise ValueError(f"unknown land-cover code {value}; expected one of {sorted(CATEGORY_LEVELS['LULC'])}")
        return value


class PredictRequest(BaseModel):
    """Score one scenario."""

    model_config = ConfigDict(extra="forbid")

    factors: FactorValues
    explain: bool = Field(True, description="Also return per-row SHAP attributions.")


class SensitivityRequest(BaseModel):
    """Sweep one factor across its training range, holding the rest fixed."""

    model_config = ConfigDict(extra="forbid")

    factors: FactorValues
    factor: FactorName
    points: int = Field(31, ge=5, le=81)
