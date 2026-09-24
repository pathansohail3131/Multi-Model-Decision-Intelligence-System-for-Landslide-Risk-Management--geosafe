"""Health endpoint.

Reports real loading progress. The simulator's startup screen polls this and
only proceeds when ``status`` is ``"ready"``, so it must answer immediately
even while the bundles are still loading on the background thread.
"""

from __future__ import annotations

from fastapi import APIRouter

from api.app.dependencies import DistrictServiceDep, PredictionServiceDep

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health")
def health(prediction: PredictionServiceDep, district: DistrictServiceDep) -> dict:
    """Loading state of the model bundles and the district map."""
    return {**prediction.health(), "district_map": district.status}
