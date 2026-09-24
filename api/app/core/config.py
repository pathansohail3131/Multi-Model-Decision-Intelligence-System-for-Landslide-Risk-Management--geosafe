"""API settings.

Everything here is either a constant or read from the environment. The data
root, model paths and logging format stay in ``config/config.yaml`` and are
resolved by :mod:`src.config`; this module only covers how the service is
exposed over HTTP.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from src.config import project_root

__all__ = ["ApiSettings", "settings"]

#: Origins allowed to call the API from a browser. The Vite dev server runs on
#: 5173; the built frontend is same-origin and needs no entry.
DEFAULT_CORS_ORIGINS = ("http://localhost:5173", "http://127.0.0.1:5173")


def _origins_from_env() -> tuple[str, ...]:
    raw = os.environ.get("LANDSLIDE_API_CORS_ORIGINS", "").strip()
    if not raw:
        return DEFAULT_CORS_ORIGINS
    return tuple(origin.strip() for origin in raw.split(",") if origin.strip())


@dataclass(frozen=True)
class ApiSettings:
    """Runtime settings for the HTTP layer."""

    title: str = "Landslide Susceptibility Simulator API"
    version: str = "1.0.0"
    description: str = (
        "Scores scenarios with the trained thematic and full-feature CatBoost bundles."
    )
    cors_origins: tuple[str, ...] = field(default_factory=_origins_from_env)

    @property
    def frontend_dist(self) -> Path:
        """Built frontend directory, served at ``/`` when it exists."""
        return project_root() / "frontend" / "dist"


settings = ApiSettings()
