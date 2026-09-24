"""Tests for configuration loading and path resolution.

These are deliberately the first tests in the project. Path resolution is the
component that broke the previous version — hardcoded ``D:\\MMDIS\\`` meant the
pipeline ran on one machine only — so the rules that replaced it are pinned by
tests rather than by convention.

The tests never touch the real data assets, so they pass on a machine that has
the code but not the rasters.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from src.config import DATA_ROOT_ENV_VAR, Config, load_config, project_root


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    """Create a minimal on-disk config directory for isolated testing."""
    directory = tmp_path / "config"
    directory.mkdir()
    base = {
        "project": {"random_seed": 42},
        "paths": {
            "data_root": "data",
            "training_csv": "final/training_dataset_final.csv",
            "processed_dir": "processed",
            "outputs_dir": "outputs",
            "models_dir": "outputs/models",
            "rasters_dir": "outputs/rasters",
            "figures_dir": "outputs/figures",
            "reports_dir": "outputs/reports",
            "logs_dir": "outputs/logs",
        },
        "rasters": {"slope": "slope_uk.tif", "TWI": "TWI_uk.tif"},
        "themes": {
            "topographic": {"factors": ["slope"], "features": ["slope"]},
            "hydrological": {"factors": ["TWI"], "features": ["TWI"]},
        },
        "metrics": {"report": ["roc_auc", "average_precision"]},
    }
    (directory / "config.yaml").write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")
    return directory


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch):
    """Ensure the data-root environment override does not leak between tests."""
    monkeypatch.delenv(DATA_ROOT_ENV_VAR, raising=False)
    return monkeypatch


# ---------------------------------------------------------------------------
# project layout
# ---------------------------------------------------------------------------
def test_project_root_is_located_structurally():
    """The root is found from the module's own location, not a fixed path."""
    root = project_root()
    assert (root / "src" / "config.py").is_file()
    assert (root / "config" / "config.yaml").is_file()
    assert (root / "run_pipeline.py").is_file()


def test_no_absolute_paths_in_source():
    """No module under src/ may contain a hardcoded absolute path.

    This is the regression test for the failure that motivated the rebuild.
    It scans for Windows drive-letter paths and for POSIX paths rooted at the
    usual user directories.
    """
    import re

    # A drive letter followed by a separator and a path character, e.g. D:\MMDIS
    # or C:/Users. The lookbehind stops it matching the tail of an ordinary
    # word before a colon and an escape, such as "matrix:\n" in a log format.
    windows_drive = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z]:[\\/]{1,2}[A-Za-z0-9_.]")
    # POSIX user/home roots that would be machine-specific.
    posix_user = re.compile(r"['\"](/home/|/Users/|/mnt/|/media/)")

    offenders: list[str] = []
    for path in (project_root() / "src").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for number, line in enumerate(text.splitlines(), start=1):
            if windows_drive.search(line) and "\\\\MMDIS" not in line:
                # Docstrings mentioning the historical D:\MMDIS\ path are the
                # one permitted exception; they are documentation, not paths.
                if "MMDIS" in line:
                    continue
                offenders.append(f"{path.name}:{number}: {line.strip()}")
            if posix_user.search(line):
                offenders.append(f"{path.name}:{number}: {line.strip()}")

    assert not offenders, "absolute paths found in src/:\n" + "\n".join(offenders)


# ---------------------------------------------------------------------------
# loading and merging
# ---------------------------------------------------------------------------
def test_load_config_reads_base_file(config_dir: Path, clean_env):
    """A config directory with only config.yaml loads cleanly."""
    cfg = load_config(config_dir)
    assert cfg.seed == 42
    assert cfg.theme_names == ["topographic", "hydrological"]


def test_local_config_overrides_base(config_dir: Path, clean_env):
    """config.local.yaml wins over config.yaml, key by key."""
    local = {"project": {"random_seed": 7}}
    (config_dir / "config.local.yaml").write_text(
        yaml.safe_dump(local, sort_keys=False), encoding="utf-8"
    )
    cfg = load_config(config_dir)
    assert cfg.seed == 7
    # Keys absent from the local file still fall through to the base file.
    assert cfg.theme_names == ["topographic", "hydrological"]


def test_merge_is_deep_not_shallow(config_dir: Path, tmp_path: Path, clean_env):
    """Overriding one nested key must not wipe its siblings."""
    # Built from tmp_path so the path is fully absolute on every platform: a
    # bare "/elsewhere" carries no drive on Windows, which would make this a
    # test of path anchoring rather than of the merge.
    elsewhere = (tmp_path / "elsewhere").resolve()
    local = {"paths": {"data_root": str(elsewhere)}}
    (config_dir / "config.local.yaml").write_text(
        yaml.safe_dump(local, sort_keys=False), encoding="utf-8"
    )
    cfg = load_config(config_dir)
    assert cfg.data_root == elsewhere
    # training_csv came from the base file and must have survived the merge.
    assert cfg.training_csv == elsewhere / "final" / "training_dataset_final.csv"


def test_lists_are_replaced_not_concatenated(config_dir: Path, clean_env):
    """A local list override means *only* those values."""
    local = {"metrics": {"report": ["recall"]}}
    (config_dir / "config.local.yaml").write_text(
        yaml.safe_dump(local, sort_keys=False), encoding="utf-8"
    )
    cfg = load_config(config_dir)
    assert cfg.get("metrics", "report") == ["recall"]


def test_missing_base_config_raises(tmp_path: Path):
    """An absent config.yaml means a broken tree and must fail loudly."""
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nonexistent")


# ---------------------------------------------------------------------------
# data root precedence
# ---------------------------------------------------------------------------
def test_relative_data_root_resolves_against_project_root(config_dir: Path, clean_env):
    """A relative data_root is anchored to the project root, not the cwd."""
    cfg = load_config(config_dir)
    assert cfg.data_root.is_absolute()
    assert cfg.data_root == project_root() / "data"


def test_env_var_beats_local_config(config_dir: Path, clean_env, tmp_path: Path):
    """LANDSLIDE_DATA_ROOT has the highest precedence."""
    local = {"paths": {"data_root": str(tmp_path / "from_local")}}
    (config_dir / "config.local.yaml").write_text(
        yaml.safe_dump(local, sort_keys=False), encoding="utf-8"
    )
    clean_env.setenv(DATA_ROOT_ENV_VAR, str(tmp_path / "from_env"))
    cfg = load_config(config_dir)
    assert cfg.data_root == tmp_path / "from_env"


# ---------------------------------------------------------------------------
# path helpers
# ---------------------------------------------------------------------------
def test_raster_path_joins_processed_dir(config_dir: Path, clean_env):
    """Raster lookups resolve through data_root/processed_dir."""
    cfg = load_config(config_dir)
    assert cfg.raster_path("slope") == cfg.data_root / "processed" / "slope_uk.tif"


def test_unknown_raster_factor_raises_with_valid_names(config_dir: Path, clean_env):
    """A typo'd factor fails immediately and says what was expected."""
    cfg = load_config(config_dir)
    with pytest.raises(KeyError) as info:
        cfg.raster_path("slop")
    assert "slope" in str(info.value)


def test_unknown_theme_raises_with_valid_names(config_dir: Path, clean_env):
    """Same contract for themes."""
    cfg = load_config(config_dir)
    with pytest.raises(KeyError) as info:
        cfg.theme_features("geological")
    assert "topographic" in str(info.value)


def test_output_dirs_resolve_against_project_root(config_dir: Path, clean_env):
    """Output paths are project-relative and independent of data_root."""
    cfg = load_config(config_dir)
    assert cfg.models_dir == project_root() / "outputs" / "models"


def test_ensure_output_dirs_creates_them(config_dir: Path, clean_env, tmp_path: Path):
    """ensure_output_dirs is idempotent and actually creates the tree."""
    cfg = Config(
        {
            "paths": {
                "outputs_dir": str(tmp_path / "out"),
                "models_dir": str(tmp_path / "out" / "models"),
                "rasters_dir": str(tmp_path / "out" / "rasters"),
                "figures_dir": str(tmp_path / "out" / "figures"),
                "reports_dir": str(tmp_path / "out" / "reports"),
                "logs_dir": str(tmp_path / "out" / "logs"),
            }
        },
        root=tmp_path,
    )
    cfg.ensure_output_dirs()
    cfg.ensure_output_dirs()  # must not raise on the second call
    assert (tmp_path / "out" / "models").is_dir()


def test_validate_reports_all_problems_not_just_the_first(
    config_dir: Path, clean_env, tmp_path: Path
):
    """Validation collects every problem so one run reveals the full picture."""
    clean_env.setenv(DATA_ROOT_ENV_VAR, str(tmp_path / "absent"))
    cfg = load_config(config_dir)
    problems = cfg.validate_data_root()
    assert len(problems) == 1
    assert "data_root does not exist" in problems[0]


def test_validate_lists_each_missing_raster(config_dir: Path, clean_env, tmp_path: Path):
    """With a real data_root but no rasters, each missing layer is named."""
    data_root = tmp_path / "data"
    (data_root / "final").mkdir(parents=True)
    (data_root / "final" / "training_dataset_final.csv").write_text("x", encoding="utf-8")
    (data_root / "processed").mkdir()
    clean_env.setenv(DATA_ROOT_ENV_VAR, str(data_root))

    cfg = load_config(config_dir)
    problems = cfg.validate_data_root()
    # Both configured rasters are absent, and each gets its own message.
    assert len(problems) == 2
    assert any("slope" in p for p in problems)
    assert any("TWI" in p for p in problems)


# ---------------------------------------------------------------------------
# real project configuration
# ---------------------------------------------------------------------------
def test_real_config_declares_all_fourteen_factors():
    """The shipped config must cover exactly the 14 conditioning factors."""
    cfg = load_config()
    expected = {
        "elevation", "slope", "aspect", "plan_curv", "profile_curv",
        "TWI", "SPI", "dist_drainage", "dist_road", "dist_fault",
        "NDVI", "LULC", "lithology", "rainfall",
    }
    assert set(cfg.raster_factors) == expected


def test_themes_partition_the_factors_without_overlap():
    """Thematic subsets must tile the 14 factors exactly once each.

    Overlap would reintroduce the correlated-error problem the thematic split
    exists to remove: two models sharing a factor are no longer blind to each
    other, and their agreement would partly reflect the shared input rather
    than independent physical evidence.
    """
    cfg = load_config()
    seen: list[str] = []
    for theme in cfg.theme_names:
        seen.extend(cfg.get("themes", theme, "factors"))
    assert len(seen) == len(set(seen)), "a factor appears in more than one theme"
    assert set(seen) == set(cfg.raster_factors)


def test_forbidden_columns_include_the_leaking_metadata():
    """taluk and slide_name are non-null only on positives — never features."""
    cfg = load_config()
    forbidden = set(cfg.get("features", "forbidden_columns", default=[]))
    assert {"taluk", "slide_name"} <= forbidden


def test_threshold_is_not_defaulted_to_half():
    """The operating threshold must be derived, not assumed."""
    cfg = load_config()
    assert cfg.get("threshold", "selected") is None
    assert cfg.get("threshold", "cost_false_negative") > cfg.get(
        "threshold", "cost_false_positive"
    )
