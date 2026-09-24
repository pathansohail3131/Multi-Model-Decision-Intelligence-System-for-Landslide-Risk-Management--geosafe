"""HTTP routes, one module per resource group."""

from api.app.routers.district import router as district_router
from api.app.routers.health import router as health_router
from api.app.routers.prediction import router as prediction_router

__all__ = ["district_router", "health_router", "prediction_router"]
