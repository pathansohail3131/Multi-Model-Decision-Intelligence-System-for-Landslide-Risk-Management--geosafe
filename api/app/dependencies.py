"""Shared service instances and the FastAPI dependencies that expose them.

The services hold the loaded model bundles, so they are process-wide
singletons: rebuilding them per request would reload five CatBoost models.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends

from api.app.services.district import DistrictService
from api.app.services.prediction import PredictionService
from src.config import load_config

__all__ = [
    "cfg",
    "prediction_service",
    "district_service",
    "get_prediction_service",
    "get_district_service",
    "PredictionServiceDep",
    "DistrictServiceDep",
]

cfg = load_config()
prediction_service = PredictionService(cfg)
district_service = DistrictService(cfg, prediction_service)


def get_prediction_service() -> PredictionService:
    return prediction_service


def get_district_service() -> DistrictService:
    return district_service


PredictionServiceDep = Annotated[PredictionService, Depends(get_prediction_service)]
DistrictServiceDep = Annotated[DistrictService, Depends(get_district_service)]
