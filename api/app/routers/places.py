"""Place-name search for the simulator's search bars. Display only."""

from __future__ import annotations

from fastapi import APIRouter, Query

from api.app.services.places import get_place_index

router = APIRouter(prefix="/api/places", tags=["places"])


@router.get("/search")
def search_places(q: str = Query("", max_length=80), limit: int = Query(8, ge=1, le=20)) -> list[dict]:
    """Villages, towns and taluks matching a name, best first."""
    places = get_place_index()
    return places.search(q, limit) if places else []
