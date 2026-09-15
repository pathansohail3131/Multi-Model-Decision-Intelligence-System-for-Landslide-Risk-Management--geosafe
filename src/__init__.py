"""Landslide susceptibility and decision-intelligence pipeline for Uttara Kannada.

The package is organised by pipeline responsibility rather than by file type:

``src.config``
    Configuration loading and path resolution. The only module that knows
    where anything lives.
``src.data``
    Loading and validating the delivered data assets, plus the reproducible
    data audit.
``src.features``
    Encoding conditioning factors into model features and splitting them into
    thematic subsets.
``src.models``
    Thematic and benchmark estimators, combination rules, and persistence that
    keeps preprocessing welded to each model.
``src.evaluation``
    Spatial and random cross-validation, the metric suite, and the ablation
    study.
``src.inference``
    Windowed district-wide raster prediction.
``src.decision``
    The layers above susceptibility: rainfall thresholds, exposure, and
    planner prioritisation.
``src.visualisation``
    Figures and cartographic output.
``src.utils``
    Seeding and logging.

Notebooks under ``notebooks/`` are for exploration only and are never imported
by the pipeline.
"""

__version__ = "0.1.0"
