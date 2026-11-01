"""FastAPI application entry point.

Run from the project root::

    python -m uvicorn api.app.main:app --host 127.0.0.1 --port 8000

When ``frontend/dist`` exists (after ``npm run build``), the built simulator is
served from ``/`` as well, so one process is enough for a demo.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.app.core.config import settings
from api.app.core.errors import Utf8JSONResponse, register_exception_handlers
from api.app.dependencies import cfg, district_service, prediction_service
from api.app.routers import district_router, health_router, places_router, prediction_router
from src.utils.logging_setup import configure_logging, get_logger

__all__ = ["app", "create_app"]

logger = get_logger("api")


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Start model loading in the background, so health answers immediately."""
    configure_logging(
        level=str(cfg.get("logging", "level", default="INFO")),
        run_name="api",
        fmt=cfg.get("logging", "format"),
        date_format=cfg.get("logging", "date_format"),
        to_file=False,
    )
    prediction_service.start()
    district_service.start()
    yield


def create_app() -> FastAPI:
    """Build the application: middleware, handlers, routes, static frontend."""
    application = FastAPI(
        title=settings.title,
        version=settings.version,
        description=settings.description,
        lifespan=lifespan,
        default_response_class=Utf8JSONResponse,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    register_exception_handlers(application)

    application.include_router(health_router)
    application.include_router(prediction_router)
    application.include_router(district_router)
    application.include_router(places_router)

    # Mounted last: StaticFiles at "/" would otherwise shadow the API routes.
    if settings.frontend_dist.exists():
        application.mount(
            "/", StaticFiles(directory=settings.frontend_dist, html=True), name="simulator"
        )
    else:
        logger.info(
            "frontend/dist not found; serving the API only. Build it with: cd frontend && npm run build"
        )

    return application


app = create_app()
