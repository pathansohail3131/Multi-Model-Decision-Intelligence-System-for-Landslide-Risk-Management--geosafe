"""Configuration loading and path resolution.

This module is the single place where the project learns where anything lives.
Nothing under ``src/`` may contain an absolute path; everything is resolved
either relative to the project root or relative to a single configurable
``data_root``.

Why this matters
----------------
The previous version of this project hardcoded ``D:\\MMDIS\\`` inside notebook
cells. The pipeline therefore ran on exactly one machine, and when that machine
changed, every path broke at once with no single place to fix them. The design
here is deliberately boring:

* ``config/config.yaml`` holds everything portable and is tracked in git.
* ``config/config.local.yaml`` holds machine-specific values and is gitignored.
* ``LANDSLIDE_DATA_ROOT`` overrides the data root for one-off runs and CI.

Resolution order for ``data_root``, first match wins:

1. the ``LANDSLIDE_DATA_ROOT`` environment variable
2. ``paths.data_root`` in ``config/config.local.yaml``
3. ``paths.data_root`` in ``config/config.yaml`` (relative to project root)

Typical use::

    from src.config import load_config

    cfg = load_config()
    df = pd.read_csv(cfg.training_csv)
    raster = cfg.raster_path("slope")
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Mapping

import yaml

__all__ = ["Config", "load_config", "project_root"]

#: Environment variable that overrides ``paths.data_root`` from any config file.
DATA_ROOT_ENV_VAR = "LANDSLIDE_DATA_ROOT"

#: Filenames looked for inside ``config/``, applied in order (later wins).
_CONFIG_FILENAMES = ("config.yaml", "config.local.yaml")


def project_root() -> Path:
    """Return the project root directory.

    The root is located structurally rather than by a hardcoded path: this file
    lives at ``<root>/src/config.py``, so the root is two levels up. This keeps
    working whether the project is run from the root, from ``notebooks/``, or
    imported from an installed location.

    Returns
    -------
    Path
        Absolute path to the project root.
    """
    return Path(__file__).resolve().parent.parent


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict:
    """Recursively merge ``override`` into ``base`` without mutating either.

    Nested mappings are merged key by key; every other type (including lists)
    is replaced wholesale. Lists are replaced rather than concatenated because
    a local override saying ``report: [roc_auc]`` should mean *only* that
    metric, not that metric appended to the defaults.

    Parameters
    ----------
    base : Mapping[str, Any]
        The lower-priority mapping.
    override : Mapping[str, Any]
        The higher-priority mapping. Its values win on conflict.

    Returns
    -------
    dict
        A new merged dictionary.
    """
    merged = copy.deepcopy(dict(base))
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], Mapping)
            and isinstance(value, Mapping)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


class Config:
    """Resolved project configuration with path helpers.

    Wraps the merged configuration dictionary and exposes the paths the
    pipeline actually asks for as properties, so that no caller ever has to
    join path fragments itself. Raw values remain reachable via
    :meth:`get` and :meth:`__getitem__` for anything not covered by a property.

    Parameters
    ----------
    data : dict
        The merged configuration mapping.
    root : Path
        Absolute path to the project root.

    Attributes
    ----------
    root : Path
        Absolute path to the project root.
    """

    def __init__(self, data: dict, root: Path) -> None:
        self._data = data
        self.root = root
        self._data_root = self._resolve_data_root()

    # ------------------------------------------------------------------
    # raw access
    # ------------------------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def get(self, *keys: str, default: Any = None) -> Any:
        """Fetch a nested value by successive keys.

        ``cfg.get("cross_validation", "n_splits")`` is equivalent to
        ``cfg["cross_validation"]["n_splits"]`` but returns ``default``
        instead of raising if any level is missing.

        Parameters
        ----------
        *keys : str
            Successive mapping keys to descend.
        default : Any, optional
            Value returned when the path does not exist.

        Returns
        -------
        Any
            The value found, or ``default``.
        """
        node: Any = self._data
        for key in keys:
            if not isinstance(node, Mapping) or key not in node:
                return default
            node = node[key]
        return node

    def as_dict(self) -> dict:
        """Return a deep copy of the merged configuration.

        Returns
        -------
        dict
            The full configuration, safe to mutate.
        """
        return copy.deepcopy(self._data)

    # ------------------------------------------------------------------
    # data paths
    # ------------------------------------------------------------------
    def _resolve_data_root(self) -> Path:
        """Resolve ``data_root`` using the documented precedence order.

        Returns
        -------
        Path
            Absolute path to the data root. The path is *not* required to
            exist here; :meth:`validate_data_root` reports on that separately
            so that import never fails merely because a drive is unplugged.
        """
        env_value = os.environ.get(DATA_ROOT_ENV_VAR)
        raw = env_value if env_value else self.get("paths", "data_root", default="data")
        candidate = Path(str(raw)).expanduser()
        # A relative data_root is interpreted against the project root, never
        # against the current working directory, so the pipeline behaves the
        # same when launched from notebooks/ as from the root.
        return candidate if candidate.is_absolute() else (self.root / candidate)

    @property
    def data_root(self) -> Path:
        """Absolute path to the data assets directory."""
        return self._data_root

    def data_path(self, *relative: str) -> Path:
        """Join path fragments onto :attr:`data_root`.

        Parameters
        ----------
        *relative : str
            Path fragments relative to the data root.

        Returns
        -------
        Path
            The joined absolute path.
        """
        return self._data_root.joinpath(*relative)

    def _configured_data_path(self, key: str) -> Path:
        """Resolve a ``paths.<key>`` entry against :attr:`data_root`."""
        value = self.get("paths", key)
        if value is None:
            raise KeyError(f"paths.{key} is not set in the configuration")
        return self.data_path(str(value))

    @property
    def training_csv(self) -> Path:
        """Path to the labelled training dataset CSV."""
        return self._configured_data_path("training_csv")

    @property
    def processed_dir(self) -> Path:
        """Directory holding the aligned conditioning-factor rasters."""
        return self._configured_data_path("processed_dir")

    @property
    def boundary_shapefile(self) -> Path:
        """Path to the district boundary shapefile."""
        return self._configured_data_path("boundary_shapefile")

    @property
    def inventory_shapefile(self) -> Path:
        """Path to the verified landslide inventory shapefile."""
        return self._configured_data_path("inventory_shapefile")

    @property
    def roads_shapefile(self) -> Path:
        """Path to the road network shapefile used for the exposure overlay."""
        return self._configured_data_path("roads_shapefile")

    @property
    def lithology_classes_json(self) -> Path:
        """Path to the lithology class-code lookup table."""
        return self._configured_data_path("lithology_classes")

    def raster_path(self, factor: str) -> Path:
        """Return the raster path for a named conditioning factor.

        Parameters
        ----------
        factor : str
            A key from the ``rasters`` block of the configuration, e.g.
            ``"slope"`` or ``"dist_road"``.

        Returns
        -------
        Path
            Absolute path to that factor's GeoTIFF.

        Raises
        ------
        KeyError
            If the factor is not declared in the configuration. Raised eagerly
            with the list of valid names, because a typo here would otherwise
            surface much later as a confusing file-not-found.
        """
        rasters = self.get("rasters", default={})
        if factor not in rasters:
            valid = ", ".join(sorted(rasters))
            raise KeyError(f"unknown raster factor {factor!r}; expected one of: {valid}")
        return self.processed_dir / str(rasters[factor])

    @property
    def raster_factors(self) -> list[str]:
        """Names of all configured conditioning-factor rasters."""
        return sorted(self.get("rasters", default={}))

    # ------------------------------------------------------------------
    # output paths
    # ------------------------------------------------------------------
    def _output_path(self, key: str) -> Path:
        """Resolve a ``paths.<key>`` output entry against the project root."""
        value = self.get("paths", key)
        if value is None:
            raise KeyError(f"paths.{key} is not set in the configuration")
        candidate = Path(str(value)).expanduser()
        return candidate if candidate.is_absolute() else (self.root / candidate)

    @property
    def outputs_dir(self) -> Path:
        """Root directory for all generated artefacts."""
        return self._output_path("outputs_dir")

    @property
    def models_dir(self) -> Path:
        """Directory for serialised model bundles."""
        return self._output_path("models_dir")

    @property
    def rasters_dir(self) -> Path:
        """Directory for generated GeoTIFF outputs."""
        return self._output_path("rasters_dir")

    @property
    def figures_dir(self) -> Path:
        """Directory for generated figures."""
        return self._output_path("figures_dir")

    @property
    def reports_dir(self) -> Path:
        """Directory for generated tables and machine-readable reports."""
        return self._output_path("reports_dir")

    @property
    def logs_dir(self) -> Path:
        """Directory for run logs."""
        return self._output_path("logs_dir")

    def ensure_output_dirs(self) -> None:
        """Create every configured output directory if it does not exist.

        Called once by the pipeline entry point. Kept out of ``__init__`` so
        that merely importing the config never has filesystem side effects,
        which would make tests and dry runs messy.
        """
        for path in (
            self.outputs_dir,
            self.models_dir,
            self.rasters_dir,
            self.figures_dir,
            self.reports_dir,
            self.logs_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # convenience
    # ------------------------------------------------------------------
    @property
    def seed(self) -> int:
        """The project-wide random seed."""
        return int(self.get("project", "random_seed", default=42))

    @property
    def theme_names(self) -> list[str]:
        """Names of the thematic models, in configuration order."""
        return list(self.get("themes", default={}))

    def theme_features(self, theme: str) -> list[str]:
        """Return the post-encoding feature names for one theme.

        Parameters
        ----------
        theme : str
            A key from the ``themes`` block, e.g. ``"hydrological"``.

        Returns
        -------
        list of str
            Feature column names belonging to that theme.

        Raises
        ------
        KeyError
            If the theme is not declared in the configuration.
        """
        themes = self.get("themes", default={})
        if theme not in themes:
            valid = ", ".join(themes)
            raise KeyError(f"unknown theme {theme!r}; expected one of: {valid}")
        return list(themes[theme]["features"])

    def validate_data_root(self, require_rasters: bool = True) -> list[str]:
        """Check that the expected data assets are reachable.

        Returns a list of human-readable problems rather than raising, so the
        entry point can print all of them at once instead of making the user
        rerun after fixing each in turn.

        Parameters
        ----------
        require_rasters : bool, default True
            Whether missing conditioning-factor rasters count as problems.
            Stages 1-6 work entirely from the training CSV, so a run that only
            covers them should not be blocked by an unavailable raster. Stage 7
            onward cannot proceed without them.

        Returns
        -------
        list of str
            One message per missing asset. Empty when everything resolves.
        """
        problems: list[str] = []
        if not self.data_root.exists():
            problems.append(
                f"data_root does not exist: {self.data_root}\n"
                f"  Set it in config/config.local.yaml or export "
                f"{DATA_ROOT_ENV_VAR}."
            )
            return problems

        if not self.training_csv.exists():
            problems.append(f"training CSV not found: {self.training_csv}")
        if not require_rasters:
            return problems

        if not self.processed_dir.exists():
            problems.append(f"processed raster directory not found: {self.processed_dir}")
        else:
            for factor in self.raster_factors:
                path = self.raster_path(factor)
                if not path.exists():
                    problems.append(f"raster for {factor!r} not found: {path}")
        return problems

    def missing_rasters(self) -> list[str]:
        """Return the names of configured rasters that are not on disk.

        Returns
        -------
        list of str
            Factor names whose GeoTIFF is absent. Empty when all are present.
        """
        if not self.processed_dir.exists():
            return list(self.raster_factors)
        return [f for f in self.raster_factors if not self.raster_path(f).exists()]

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"Config(root={self.root}, data_root={self.data_root}, "
            f"seed={self.seed}, themes={self.theme_names})"
        )


def load_config(config_dir: Path | str | None = None) -> Config:
    """Load, merge and resolve the project configuration.

    Reads ``config/config.yaml`` then deep-merges ``config/config.local.yaml``
    over it if present. Missing local config is normal and not an error.

    Parameters
    ----------
    config_dir : Path or str, optional
        Directory containing the YAML config files. Defaults to
        ``<project_root>/config``.

    Returns
    -------
    Config
        The resolved configuration object.

    Raises
    ------
    FileNotFoundError
        If the base ``config.yaml`` is absent, which means the project tree is
        broken rather than merely unconfigured.
    """
    root = project_root()
    directory = Path(config_dir) if config_dir is not None else (root / "config")

    base_path = directory / _CONFIG_FILENAMES[0]
    if not base_path.exists():
        raise FileNotFoundError(f"base configuration not found: {base_path}")

    merged: dict = {}
    for filename in _CONFIG_FILENAMES:
        path = directory / filename
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if not isinstance(loaded, Mapping):
            raise ValueError(f"configuration file is not a mapping: {path}")
        merged = _deep_merge(merged, loaded)

    return Config(merged, root)
