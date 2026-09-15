#!/usr/bin/env python3
"""Single entry point for the landslide susceptibility and decision pipeline.

Every result in ``docs/results.md`` is produced by this script. There is no
second path through the code and no notebook that must be run first —
notebooks under ``notebooks/`` are for exploration only.

Usage
-----
Check that the environment and data are wired up correctly::

    python run_pipeline.py check

Run the whole pipeline end to end::

    python run_pipeline.py all

Run one stage, or a range::

    python run_pipeline.py stage 3
    python run_pipeline.py stage 3 --to 5

List the stages and their status::

    python run_pipeline.py stages

Stage numbering matches ``docs/methodology.md``. Stages are ordered because
each consumes the previous stage's artefacts; running a later stage without
its inputs fails with a message naming the stage you need to run first, rather
than with a stack trace.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from typing import Callable

from src.config import DATA_ROOT_ENV_VAR, load_config
from src.data.audit import run_audit
from src.evaluation.interpretability import run_shap_stage
from src.evaluation.spatial_stage import run_spatial_cv_stage
from src.decision.layers import run_decision_stage
from src.features.build import run_feature_stage
from src.inference.raster import run_inference_stage
from src.models.calibration import run_calibration_stage
from src.models.train import run_training_stage
from src.utils.logging_setup import configure_logging, get_logger
from src.visualisation.maps import run_figures
from src.utils.seeds import seed_environment_report, set_global_seed

logger = get_logger("pipeline")


@dataclass(frozen=True)
class Stage:
    """One pipeline stage.

    Attributes
    ----------
    number : int
        Stage index; also the CLI selector.
    name : str
        Short human-readable name.
    description : str
        What the stage does, shown by ``run_pipeline.py stages``.
    runner : Callable or None
        Callable taking the resolved ``Config``. ``None`` marks a stage that
        is scaffolded but not yet implemented.
    strict_determinism : bool
        Whether to pin thread pools before running. True for stages whose
        numeric output is reported and must be bit-reproducible; False for
        stages producing smooth raster surfaces where speed matters more.
    requires_rasters : bool
        Whether the stage cannot run without the conditioning-factor rasters.
        Stages 1-6 work from the training CSV alone, so an unavailable raster
        degrades their reporting rather than blocking the run; stage 7 onward
        genuinely cannot proceed.
    """

    number: int
    name: str
    description: str
    runner: Callable | None = None
    strict_determinism: bool = True
    requires_rasters: bool = False
    requires: tuple[int, ...] = field(default=())


def _not_implemented(stage: Stage) -> Callable:
    """Build a placeholder runner that fails clearly for an unbuilt stage."""

    def _runner(_cfg) -> None:
        raise NotImplementedError(
            f"Stage {stage.number} ({stage.name}) is scaffolded but not yet "
            f"implemented. The project is being built stage by stage; see "
            f"docs/methodology.md for the plan."
        )

    return _runner


#: The pipeline, in execution order. Runners are attached as each stage lands.
STAGES: tuple[Stage, ...] = (
    Stage(
        number=1,
        name="data-audit",
        description="Load and validate data assets; produce the reproducible data audit.",
        runner=run_audit,
    ),
    Stage(
        number=2,
        name="features",
        description="Encode features (circular aspect, flat flag, log1p SPI, categoricals) and define thematic groups.",
        runner=run_feature_stage,
        requires=(1,),
    ),
    Stage(
        number=3,
        name="spatial-cv",
        description="Spatial block CV vs random CV; empirical comparison of class-imbalance strategies.",
        runner=run_spatial_cv_stage,
        requires=(2,),
    ),
    Stage(
        number=4,
        name="models-ablation",
        description="Train four thematic models plus the full-feature benchmark; ablate combination rules.",
        runner=run_training_stage,
        requires=(3,),
    ),
    Stage(
        number=5,
        name="confidence-threshold",
        description="Calibrate thematic models, derive the disagreement confidence measure, select the operating threshold.",
        runner=run_calibration_stage,
        requires=(4,),
    ),
    Stage(
        number=6,
        name="interpretability",
        description="SHAP per thematic model and cross-theme agreement analysis.",
        runner=run_shap_stage,
        requires=(4,),
    ),
    Stage(
        number=7,
        name="raster-inference",
        description="Windowed district-wide inference producing the susceptibility and confidence rasters.",
        runner=run_inference_stage,
        strict_determinism=False,
        requires_rasters=False,
        requires=(5,),
    ),
    Stage(
        number=8,
        name="decision-layer",
        description="Rainfall thresholds, exposure overlay, prioritised planner output, NISAR multiplier slot.",
        runner=run_decision_stage,
        strict_determinism=False,
        requires_rasters=False,
        requires=(7,),
    ),
)

STAGES_BY_NUMBER = {stage.number: stage for stage in STAGES}


def command_check(cfg) -> int:
    """Validate configuration, data availability and the runtime environment.

    This is the first thing to run on a new machine. It reports every problem
    it finds at once rather than stopping at the first, because fixing paths
    one failed run at a time is exactly the experience this project is trying
    to avoid.

    Parameters
    ----------
    cfg : Config
        The resolved configuration.

    Returns
    -------
    int
        Process exit code: 0 when everything resolves, 1 otherwise.
    """
    logger.info("Project root : %s", cfg.root)
    logger.info("Data root    : %s", cfg.data_root)
    logger.info("Seed         : %s", cfg.seed)
    logger.info("Themes       : %s", ", ".join(cfg.theme_names))

    problems = cfg.validate_data_root()

    logger.info("--- dependency versions ---")
    missing: list[str] = []
    for module_name in (
        "numpy",
        "pandas",
        "scipy",
        "sklearn",
        "xgboost",
        "catboost",
        "shap",
        "rasterio",
        "geopandas",
        "yaml",
        "joblib",
        "matplotlib",
    ):
        try:
            module = __import__(module_name)
            logger.info("  %-12s %s", module_name, getattr(module, "__version__", "?"))
        except ImportError:
            missing.append(module_name)
            logger.error("  %-12s MISSING", module_name)

    if missing:
        problems.append(
            "missing packages: "
            + ", ".join(missing)
            + "\n  Install them with: pip install -r requirements.txt"
        )

    if problems:
        logger.error("--- %d problem(s) found ---", len(problems))
        for problem in problems:
            logger.error("  * %s", problem)
        logger.error(
            "Data paths are configured in config/config.local.yaml or via the "
            "%s environment variable.",
            DATA_ROOT_ENV_VAR,
        )
        return 1

    logger.info("--- all checks passed ---")
    return 0


def command_stages(_cfg) -> int:
    """Print the stage table with implementation status.

    Returns
    -------
    int
        Always 0.
    """
    logger.info("%-4s %-22s %-9s %s", "No.", "Name", "Status", "Description")
    for stage in STAGES:
        status = "ready" if stage.runner is not None else "pending"
        logger.info(
            "%-4d %-22s %-9s %s", stage.number, stage.name, status, stage.description
        )
    return 0


def run_stage(stage: Stage, cfg) -> None:
    """Execute a single stage with the appropriate determinism settings.

    Parameters
    ----------
    stage : Stage
        The stage to run.
    cfg : Config
        The resolved configuration.

    Raises
    ------
    NotImplementedError
        If the stage is scaffolded but not yet built.
    """
    logger.info("=" * 78)
    logger.info("STAGE %d - %s", stage.number, stage.name)
    logger.info("%s", stage.description)
    logger.info("=" * 78)

    set_global_seed(cfg.seed, strict=stage.strict_determinism)

    runner = stage.runner or _not_implemented(stage)
    runner(cfg)

    logger.info("Stage %d complete.", stage.number)


def command_run(cfg, first: int, last: int) -> int:
    """Run a contiguous range of stages.

    Parameters
    ----------
    cfg : Config
        The resolved configuration.
    first, last : int
        Inclusive stage range.

    Returns
    -------
    int
        Process exit code.
    """
    selected = [s for s in STAGES if first <= s.number <= last]
    if not selected:
        logger.error("No stages in range %d-%d.", first, last)
        return 1

    # Only demand the rasters if a selected stage actually needs them. Stages
    # 1-6 run from the training CSV, so a missing layer degrades their
    # reporting rather than blocking the run.
    needs_rasters = any(s.requires_rasters for s in selected)
    problems = cfg.validate_data_root(require_rasters=needs_rasters)
    if problems:
        logger.error("Cannot run: required data assets are not reachable.")
        for problem in problems:
            logger.error("  * %s", problem)
        logger.error("Run `python run_pipeline.py check` for details.")
        return 1

    if not needs_rasters:
        missing = cfg.missing_rasters()
        if missing:
            logger.warning(
                "%d raster(s) unavailable: %s. Stages 1-6 do not need them; "
                "stage 7 onward will.",
                len(missing),
                sorted(missing),
            )

    for stage in selected:
        try:
            run_stage(stage, cfg)
        except NotImplementedError as exc:
            logger.error("%s", exc)
            return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line parser.

    Returns
    -------
    argparse.ArgumentParser
        The configured parser.
    """
    parser = argparse.ArgumentParser(
        prog="run_pipeline.py",
        description="Landslide susceptibility and decision-intelligence pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--log-level",
        default=None,
        help="Override the configured log level (DEBUG, INFO, WARNING, ERROR).",
    )
    parser.add_argument(
        "--no-log-file",
        action="store_true",
        help="Log to the console only; do not write a run log under outputs/logs/.",
    )

    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check", help="Validate configuration, data paths and dependencies.")
    sub.add_parser("stages", help="List pipeline stages and their status.")
    sub.add_parser("all", help="Run every stage end to end.")
    sub.add_parser("figures", help="Regenerate all figures from artefacts on disk.")

    stage_parser = sub.add_parser("stage", help="Run a single stage or a range.")
    stage_parser.add_argument("number", type=int, help="Stage number to run.")
    stage_parser.add_argument(
        "--to",
        type=int,
        default=None,
        help="Run through this stage inclusive, instead of just one.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Parameters
    ----------
    argv : list of str, optional
        Argument vector; defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Process exit code.
    """
    args = build_parser().parse_args(argv)

    cfg = load_config()
    cfg.ensure_output_dirs()

    log_level = args.log_level or cfg.get("logging", "level", default="INFO")
    to_file = bool(cfg.get("logging", "to_file", default=True)) and not args.no_log_file
    log_path = configure_logging(
        level=log_level,
        log_dir=cfg.logs_dir,
        run_name=args.command,
        fmt=cfg.get("logging", "format"),
        date_format=cfg.get("logging", "date_format"),
        to_file=to_file,
    )
    if log_path is not None:
        logger.info("Run log: %s", log_path)
    logger.debug("Determinism environment: %s", seed_environment_report())

    if args.command == "check":
        return command_check(cfg)
    if args.command == "stages":
        return command_stages(cfg)
    if args.command == "figures":
        produced = run_figures(cfg)
        for name, path in produced.items():
            logger.info("  %-20s %s", name, path)
        return 0
    if args.command == "all":
        return command_run(cfg, STAGES[0].number, STAGES[-1].number)
    if args.command == "stage":
        if args.number not in STAGES_BY_NUMBER:
            valid = ", ".join(str(s.number) for s in STAGES)
            logger.error("Unknown stage %d; valid stages are: %s", args.number, valid)
            return 1
        last = args.to if args.to is not None else args.number
        return command_run(cfg, args.number, last)

    return 1


if __name__ == "__main__":
    sys.exit(main())
