"""Presentation metadata for the 14 conditioning factors.

Units and descriptions come from ``docs/data_dictionary.md``. Physical bounds
are hard validation limits — values outside them are not terrain. They are
deliberately much wider than the training range: a value outside the training
range is still accepted, but is reported back as extrapolation rather than
silently scored.
"""

from __future__ import annotations

from src.features.encoders import LITHOLOGY_LABELS, LULC_LABELS

__all__ = [
    "FACTORS",
    "CATEGORICAL",
    "CATEGORY_LEVELS",
    "LITHOLOGY_NAMES",
    "LULC_NAMES",
    "SHAP_SOURCE_FACTOR",
]

#: Human-readable names for lithology codes, keyed by the encoder's labels.
LITHOLOGY_NAMES = {
    "mt_metamorphic": "Metamorphic",
    "pa_acid_plutonic": "Acid plutonic",
    "su_siliciclastic_sedimentary": "Siliciclastic sedimentary",
    "vb_basic_volcanic": "Basic volcanic",
    "wb_water_bodies": "Water bodies",
    "unmapped_coastal_gap": "Unmapped (coastal GLiM gap)",
}

#: Human-readable names for ESA WorldCover codes, keyed by the encoder's labels.
LULC_NAMES = {
    "tree_cover": "Tree cover",
    "shrubland": "Shrubland",
    "grassland": "Grassland",
    "cropland": "Cropland",
    "built_up": "Built-up",
    "bare_sparse": "Bare / sparse vegetation",
    "snow_ice": "Snow and ice",
    "permanent_water": "Permanent water",
    "herbaceous_wetland": "Herbaceous wetland",
    "mangroves": "Mangroves",
    "moss_lichen": "Moss and lichen",
}

CATEGORICAL = ("lithology", "LULC")

#: Every factor the models consume. ``bounds`` are physical validation limits.
FACTORS: dict[str, dict] = {
    "elevation": {
        "label": "Elevation",
        "unit": "m a.s.l.",
        "bounds": [0.0, 3000.0],
        "decimals": 0,
        "scale": "linear",
        "description": "Height above sea level from the SRTM 1-arcsecond DEM.",
    },
    "slope": {
        "label": "Slope angle",
        "unit": "°",
        "bounds": [0.0, 90.0],
        "decimals": 1,
        "scale": "linear",
        "description": "Terrain gradient from the filled DEM. A slope of 0 marks the cell flat, which switches aspect off.",
    },
    "aspect": {
        "label": "Aspect",
        "unit": "° from N",
        "bounds": [0.0, 360.0],
        "decimals": 0,
        "scale": "linear",
        "description": "Compass direction the slope faces, clockwise from north. Encoded as sin/cos.",
    },
    "plan_curv": {
        "label": "Plan curvature",
        "unit": "1/m",
        "bounds": [-10.0, 10.0],
        "decimals": 4,
        "scale": "linear",
        "description": "Curvature across the slope; controls whether surface flow converges or diverges.",
    },
    "profile_curv": {
        "label": "Profile curvature",
        "unit": "1/m",
        "bounds": [-10.0, 10.0],
        "decimals": 4,
        "scale": "linear",
        "description": "Curvature along the slope; controls whether surface flow accelerates or decelerates.",
    },
    "TWI": {
        "label": "Topographic wetness index",
        "unit": "",
        "bounds": [0.0, 50.0],
        "decimals": 2,
        "scale": "linear",
        "description": "ln(catchment area / tan slope). Higher means the ground tends to stay wetter.",
    },
    "SPI": {
        "label": "Stream power index",
        "unit": "",
        "bounds": [0.0, 1.0e8],
        "decimals": 3,
        "scale": "log",
        "description": "Catchment area × tan slope: erosive power of surface flow. Spans orders of magnitude; the model uses log1p.",
    },
    "dist_drainage": {
        "label": "Distance to drainage",
        "unit": "m",
        "bounds": [0.0, 100000.0],
        "decimals": 0,
        "scale": "linear",
        "description": "Euclidean distance to the extracted stream network.",
    },
    "rainfall": {
        "label": "Annual rainfall",
        "unit": "mm/yr",
        "bounds": [0.0, 15000.0],
        "decimals": 0,
        "scale": "linear",
        "description": "CHIRPS 5-year mean (2019–2023). Known to under-estimate orographic rain; the spatial pattern is what the model uses.",
    },
    "lithology": {
        "label": "Lithology",
        "unit": "",
        "description": "GLiM rock class. 'Unmapped' marks a coastal coverage gap, not a rock type.",
    },
    "dist_fault": {
        "label": "Distance to lineament",
        "unit": "m",
        "bounds": [0.0, 100000.0],
        "decimals": 0,
        "scale": "linear",
        "description": "Distance to DEM-derived structural lineaments — a fault proxy, not mapped fault data.",
    },
    "dist_road": {
        "label": "Distance to road",
        "unit": "m",
        "bounds": [0.0, 100000.0],
        "decimals": 0,
        "scale": "linear",
        "description": "Distance to the OSM road network. Partly reflects road-cut failures, partly inventory accessibility bias.",
    },
    "LULC": {
        "label": "Land cover",
        "unit": "",
        "description": "ESA WorldCover class at the cell.",
    },
    "NDVI": {
        "label": "NDVI",
        "unit": "",
        "bounds": [-1.0, 1.0],
        "decimals": 3,
        "scale": "linear",
        "description": "Vegetation index, −1 to 1. Higher means denser, healthier vegetation.",
    },
}

#: Categorical code -> (encoder label, display name).
CATEGORY_LEVELS: dict[str, dict[int, tuple[str, str]]] = {
    "lithology": {
        -1: ("unmapped_coastal_gap", LITHOLOGY_NAMES["unmapped_coastal_gap"]),
        **{code: (label, LITHOLOGY_NAMES[label]) for code, label in LITHOLOGY_LABELS.items()},
    },
    "LULC": {code: (label, LULC_NAMES[label]) for code, label in LULC_LABELS.items()},
}

#: Encoded model feature -> the raw factor it is computed from. SHAP values are
#: additive, so summing a factor's encoded columns gives that factor's share.
#: ``is_flat`` is a function of slope alone, so it is credited to slope.
SHAP_SOURCE_FACTOR = {
    "aspect_sin": "aspect",
    "aspect_cos": "aspect",
    "is_flat": "slope",
    "SPI_log": "SPI",
}
