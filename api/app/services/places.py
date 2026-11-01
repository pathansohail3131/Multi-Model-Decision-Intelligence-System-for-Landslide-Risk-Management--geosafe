"""Place names for a coordinate: village, taluk and nearest town.

Display only. The models never see these layers and no score depends on them.
The data in ``api/app/data/places`` is built by ``scripts/build_places.py``, in
the project grid (EPSG:32643), so lookups need no reprojection.
"""

from __future__ import annotations

import json
import math
import re
import threading
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import shapely
from shapely.geometry import shape
from shapely.strtree import STRtree

from src.utils.logging_setup import get_logger

__all__ = ["PlaceIndex", "get_place_index"]

logger = get_logger(__name__)

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "places"
DISTRICT = "Uttara Kannada"
STATE = "Karnataka"
OUTSIDE_LABEL = "Outside mapped villages"

#: A point this close to a village, but outside every polygon, is "Near" it.
NEAR_VILLAGE_M = 3000.0

COMPASS_8 = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")

#: Search results below this similarity are not offered as fuzzy matches.
FUZZY_MIN_RATIO = 0.72
#: Among equally good matches: towns, then taluks, then villages.
KIND_ORDER = {"town": 0, "taluk": 1, "village": 2}


def _features(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["features"]


def _search_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _compass(dx: float, dy: float) -> str:
    bearing = math.degrees(math.atan2(dx, dy)) % 360
    return COMPASS_8[round(bearing / 45) % 8]


def _rings(geometry) -> list[list[list[float]]]:
    polygons = getattr(geometry, "geoms", [geometry])
    rings = []
    for polygon in polygons:
        for ring in (polygon.exterior, *polygon.interiors):
            rings.append([[x, y] for x, y in ring.coords])
    return rings


class PlaceIndex:
    """Spatial lookup over the village, taluk and town layers.

    Parameters
    ----------
    data_dir : Path
        Folder holding ``villages``, ``taluks`` and ``towns`` ``.geojson``.
    """

    def __init__(self, data_dir: Path = DATA_DIR) -> None:
        villages = _features(data_dir / "villages.geojson")
        taluks = _features(data_dir / "taluks.geojson")
        towns = _features(data_dir / "towns.geojson")

        self._villages = [f["properties"] for f in villages]
        self._village_tree = STRtree([shape(f["geometry"]) for f in villages])

        self._taluk_names = [f["properties"]["taluk"] for f in taluks]
        self._taluk_shapes = [shape(f["geometry"]) for f in taluks]
        self._taluk_tree = STRtree(self._taluk_shapes)

        self._town_names = [f["properties"]["name"] for f in towns]
        self._town_xy = np.array([f["geometry"]["coordinates"][:2] for f in towns], dtype="float64")
        self._search_entries = self._build_search_entries()

    def _build_search_entries(self) -> list[dict]:
        """One entry per distinct name, kind and taluk. Census towns split into
        several polygons collapse to one, and a headquarters point wins over them."""
        candidates = []
        for name, (x, y) in zip(self._town_names, self._town_xy):
            taluk = self._taluk_at(shapely.Point(x, y))
            candidates.append((name, "town", taluk, float(x), float(y)))
        for name, geometry in zip(self._taluk_names, self._taluk_shapes):
            point = geometry.representative_point()
            candidates.append((name, "taluk", name, point.x, point.y))
        for props, geometry in zip(self._villages, self._village_tree.geometries):
            point = geometry.representative_point()
            kind = "town" if props["town"] else "village"
            candidates.append((props["village"], kind, props["taluk"], point.x, point.y))

        entries: dict[tuple, dict] = {}
        for name, kind, taluk, x, y in candidates:
            key = _search_key(name)
            entries.setdefault(
                (key, kind, taluk),
                {"key": key, "name": name, "kind": kind, "taluk": taluk, "x": round(x, 1), "y": round(y, 1)},
            )
        return list(entries.values())

    def search(self, query: str, limit: int = 8) -> list[dict]:
        """Find places by name: exact, starts-with, contains, then fuzzy.

        Case, spaces and punctuation are ignored. Queries under 2 characters
        return nothing.
        """
        q = _search_key(query)
        if len(q) < 2:
            return []

        def order(entry: dict) -> tuple:
            return len(entry["key"]), KIND_ORDER[entry["kind"]], entry["name"]

        tiers: list[list[dict]] = [[], [], []]
        rest = []
        for entry in self._search_entries:
            key = entry["key"]
            if key == q:
                tiers[0].append(entry)
            elif key.startswith(q):
                tiers[1].append(entry)
            elif q in key:
                tiers[2].append(entry)
            else:
                rest.append(entry)
        ranked = [entry for tier in tiers for entry in sorted(tier, key=order)]

        if len(ranked) < limit:
            scored = []
            for entry in rest:
                ratio = SequenceMatcher(None, q, entry["key"]).ratio()
                if ratio >= FUZZY_MIN_RATIO:
                    scored.append((ratio, entry))
            scored.sort(key=lambda item: (-item[0], *order(item[1])))
            ranked.extend(entry for _, entry in scored)

        return [{k: e[k] for k in ("name", "kind", "taluk", "x", "y")} for e in ranked[:limit]]

    def _nearest_town(self, x: float, y: float) -> dict | None:
        if not len(self._town_xy):
            return None
        dx = x - self._town_xy[:, 0]
        dy = y - self._town_xy[:, 1]
        distances = np.hypot(dx, dy)
        i = int(distances.argmin())
        return {
            "name": self._town_names[i],
            "distance_m": round(float(distances[i]), 1),
            "direction_from_town": _compass(float(dx[i]), float(dy[i])),
        }

    def _taluk_at(self, point) -> str | None:
        hits = self._taluk_tree.query(point, predicate="intersects")
        return self._taluk_names[int(hits.min())] if len(hits) else None

    def lookup(self, x: float, y: float) -> dict:
        """Name a projected coordinate (EPSG:32643)."""
        village = None
        inside = False
        is_town = False
        taluk = None
        nearest_town = None
        label = OUTSIDE_LABEL

        if math.isfinite(x) and math.isfinite(y):
            point = shapely.Point(x, y)
            nearest_town = self._nearest_town(x, y)
            hits = self._village_tree.query(point, predicate="intersects")
            if len(hits):
                props = self._villages[int(hits.min())]
                village, taluk, is_town, inside = props["village"], props["taluk"], bool(props["town"]), True
                label = f"{village} town" if is_town else village
            else:
                taluk = self._taluk_at(point)
                i = int(self._village_tree.nearest(point))
                if self._village_tree.geometries[i].distance(point) <= NEAR_VILLAGE_M:
                    props = self._villages[i]
                    village = props["village"]
                    taluk = taluk or props["taluk"]
                    label = f"Near {village}"
                elif taluk:
                    label = f"{taluk} taluk"

        return {
            "village": village,
            "inside_village": inside,
            "is_town": is_town,
            "taluk": taluk,
            "district": DISTRICT,
            "state": STATE,
            "nearest_town": nearest_town,
            "label": label,
            "sublabel": f"{taluk} taluk, {DISTRICT}" if taluk else DISTRICT,
        }

    def taluk_outlines(self) -> list[dict]:
        """Taluk rings and a label point. Holes are rings too: fill with even-odd."""
        outlines = []
        for name, geometry in zip(self._taluk_names, self._taluk_shapes):
            label = geometry.representative_point()
            outlines.append({"name": name, "rings": _rings(geometry), "label": [label.x, label.y]})
        return outlines


_lock = threading.Lock()
_index: PlaceIndex | None = None
_loaded = False


def get_place_index() -> PlaceIndex | None:
    """The shared index, built on first use. ``None`` when the data cannot load."""
    global _index, _loaded
    if _loaded:
        return _index
    with _lock:
        if not _loaded:
            try:
                _index = PlaceIndex()
            except Exception:  # noqa: BLE001 - place names are a display nicety
                logger.exception("Place names unavailable; responses will omit them")
            _loaded = True
    return _index
