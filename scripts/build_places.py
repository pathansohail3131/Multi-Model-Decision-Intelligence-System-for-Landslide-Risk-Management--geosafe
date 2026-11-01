"""Build the place-name layers the API uses to name a location.

One-off: run it again only to refresh ``api/app/data/places/``. The API reads
the output without geopandas, and no model or pipeline stage touches it.

Fetch the source files first (MIT licence, three files only)::

    git clone --depth 1 --filter=blob:none --sparse --no-checkout \
        https://github.com/datta07/INDIAN-SHAPEFILES indian-shapefiles
    cd indian-shapefiles
    git sparse-checkout set --no-cone "/STATES/KARNATAKA/KARNATAKA_VILLAGES.geojson" \
        "/STATES/KARNATAKA/KARNATAKA_SUBDISTRICTS.geojson" \
        "/STATES/KARNATAKA/KARNATAKA Sub District Hq.geojson"
    git checkout

Then, from the project root::

    python scripts/build_places.py path/to/indian-shapefiles/STATES/KARNATAKA
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import geopandas as gpd
import shapely
from shapely.geometry import mapping

#: Census 2011 district code of Uttara Kannada.
DISTRICT_CODE = "563"
#: The project grid (config/config.yaml, grid.crs).
TARGET_CRS = "EPSG:32643"
VILLAGE_TOLERANCE_M = 20
TALUK_TOLERANCE_M = 40

#: Census town suffixes: "Sirsi (CMC + OG)", "Ankola (TP)".
TOWN_SUFFIX = re.compile(r"\s*\((?:CMC|TMC|TP|CT|NAC|M|M Corp\.?)(?:\s*\+\s*OG)?\)\s*$", re.IGNORECASE)
TALUK_RENAMES = {"Supa": "Joida (Supa)"}

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "api" / "app" / "data" / "places"


def _clean(value) -> str:
    if not isinstance(value, str):
        return ""
    value = re.sub(r"\s+", " ", value).strip()
    return re.sub(r"\(\s*(.*?)\s*\)", lambda match: f"({match.group(1)})", value)


def _taluk(name) -> str:
    name = _clean(name)
    return TALUK_RENAMES.get(name, name)


def _village_name(row) -> tuple[str, bool]:
    name = _clean(row["VILNAME11"]) or _clean(row["VILNAM_SOI"]).title()
    stripped = TOWN_SUFFIX.sub("", name)
    return stripped, stripped != name


def _read(path: Path, code_field: str) -> gpd.GeoDataFrame:
    frame = gpd.read_file(path)
    if frame.crs is None:
        frame = frame.set_crs("EPSG:4326")
    return frame[frame[code_field].astype(str) == DISTRICT_CODE].to_crs(TARGET_CRS)


def _simplified(frame: gpd.GeoDataFrame, tolerance: float):
    return shapely.make_valid(frame.geometry.simplify(tolerance, preserve_topology=True).values)


def _write(path: Path, geometries, properties: list[dict]) -> None:
    features = []
    for geometry, props in zip(geometries, properties):
        rounded = shapely.set_precision(geometry, 1.0)
        # Source slivers under 1 m2 collapse at 1 m precision; there is nothing to draw or hit.
        if rounded.is_empty:
            print(f"dropped, collapses at 1 m: {props}")
            continue
        features.append({"type": "Feature", "properties": props, "geometry": _as_ints(mapping(rounded))})
    collection = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": TARGET_CRS}},
        "features": features,
    }
    path.write_text(json.dumps(collection, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
    print(f"{path.name}: {len(features)} features, {path.stat().st_size / 1024:.0f} KB")


def _as_ints(value):
    if isinstance(value, dict):
        return {key: _as_ints(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_ints(item) for item in value]
    return int(round(value)) if isinstance(value, float) else value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", type=Path, help="The STATES/KARNATAKA folder of INDIAN-SHAPEFILES")
    source = parser.parse_args().source
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    villages = _read(source / "KARNATAKA_VILLAGES.geojson", "DTCODE11")
    properties = []
    for _, row in villages.iterrows():
        name, town = _village_name(row)
        properties.append({"village": name, "taluk": _taluk(row["SDTNAME"]), "town": town})
    _write(OUTPUT_DIR / "villages.geojson", _simplified(villages, VILLAGE_TOLERANCE_M), properties)

    taluks = _read(source / "KARNATAKA_SUBDISTRICTS.geojson", "dtcode11")
    _write(
        OUTPUT_DIR / "taluks.geojson",
        _simplified(taluks, TALUK_TOLERANCE_M),
        [{"taluk": _taluk(name)} for name in taluks["sdtname"]],
    )

    towns = _read(source / "KARNATAKA Sub District Hq.geojson", "DTCODE11")
    _write(OUTPUT_DIR / "towns.geojson", towns.geometry.values, [{"name": _clean(n)} for n in towns["NAME11"]])


if __name__ == "__main__":
    main()
