"""District endpoints: map thumbnails and raster sampling at a coordinate."""

from __future__ import annotations

from fastapi import APIRouter, Query
from fastapi.responses import Response

from api.app.core.errors import error_response
from api.app.dependencies import DistrictServiceDep

router = APIRouter(prefix="/api/district", tags=["district"])


@router.get("/meta")
def district_meta(service: DistrictServiceDep) -> dict:
    """Grid, available layers and the recorded-landslide inventory."""
    return service.meta()


@router.get("/image/{layer}.png")
def district_image(layer: str, service: DistrictServiceDep):
    """Return a rendered district layer as PNG."""
    data = service.image(layer)
    if data is None:
        if service.status == "loading":
            return error_response(503, "map_loading", "The district map is still being prepared.")
        return error_response(404, "layer_not_found", f"No district layer named {layer!r}.")
    return Response(content=data, media_type="image/png", headers={"Cache-Control": "max-age=300"})


@router.get("/sample")
def district_sample(service: DistrictServiceDep, x: float = Query(...), y: float = Query(...)) -> dict:
    """Read the 14 conditioning factors from the rasters at a projected coordinate."""
    return service.sample(x, y)
