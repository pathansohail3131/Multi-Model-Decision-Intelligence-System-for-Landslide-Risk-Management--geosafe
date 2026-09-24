"""Scoring endpoints: metadata, prediction, sensitivity sweeps, sample scenarios."""

from __future__ import annotations

from fastapi import APIRouter, Query

from api.app.dependencies import PredictionServiceDep
from api.app.schemas.requests import PredictRequest, SensitivityRequest

router = APIRouter(prefix="/api", tags=["prediction"])


@router.get("/metadata")
def metadata(service: PredictionServiceDep) -> dict:
    """Factor specifications, per-model metrics, threshold, classes and caveats."""
    return service.metadata()


@router.post("/predict")
def predict(request: PredictRequest, service: PredictionServiceDep) -> dict:
    """Score one 30 m cell with the four thematic models and the full-feature model."""
    return service.predict(request.factors.model_dump(), explain=request.explain)


@router.post("/sensitivity")
def sensitivity(request: SensitivityRequest, service: PredictionServiceDep) -> dict:
    """Re-score the scenario across one factor's training distribution."""
    return service.sensitivity(request.factors.model_dump(), request.factor, request.points)


@router.get("/samples/random")
def random_sample(service: PredictionServiceDep, label: int | None = Query(None, ge=0, le=1)) -> dict:
    """Return one real training row, flagged as already seen by the models."""
    return service.training_sample(label=label)
