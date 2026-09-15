"""Logging configuration for pipeline runs.

Every stage logs to the console and, by default, to a timestamped file under
``outputs/logs/``. The file log is what makes a result auditable months later:
it records which configuration was active, which data root was resolved, how
many samples were loaded and which decisions the pipeline took, so that a
number in ``docs/results.md`` can be traced back to the run that produced it.

Log records deliberately carry the module name. When something looks wrong in
a susceptibility map, the first question is always *which stage produced this*,
and a bare message cannot answer it.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

__all__ = ["configure_logging", "get_logger"]

#: Guard so repeated calls (e.g. from notebooks) do not stack duplicate
#: handlers and print every line two or three times.
_CONFIGURED = False


def configure_logging(
    level: str = "INFO",
    log_dir: Path | None = None,
    run_name: str = "pipeline",
    fmt: str | None = None,
    date_format: str | None = None,
    to_file: bool = True,
) -> Path | None:
    """Set up root logging for a pipeline run.

    Parameters
    ----------
    level : str, default "INFO"
        Logging level name applied to the root logger.
    log_dir : Path, optional
        Directory for the run log file. Created if absent. Required when
        ``to_file`` is True.
    run_name : str, default "pipeline"
        Stem for the log filename; a UTC timestamp is appended so that
        successive runs never overwrite one another.
    fmt : str, optional
        Log record format string. Falls back to a sensible default.
    date_format : str, optional
        Timestamp format for log records.
    to_file : bool, default True
        Whether to additionally write the log to a file.

    Returns
    -------
    Path or None
        Path to the log file, or None when file logging is disabled.

    Notes
    -----
    Calling this more than once is safe: subsequent calls are ignored rather
    than adding duplicate handlers.
    """
    global _CONFIGURED

    fmt = fmt or "%(asctime)s | %(levelname)-8s | %(name)-28s | %(message)s"
    date_format = date_format or "%Y-%m-%d %H:%M:%S"

    if _CONFIGURED:
        return None

    formatter = logging.Formatter(fmt=fmt, datefmt=date_format)
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Clear any handlers installed by an importing environment (Jupyter does
    # this) so output is not duplicated.
    for handler in list(root.handlers):
        root.removeHandler(handler)

    console = logging.StreamHandler(stream=sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)

    log_path: Path | None = None
    if to_file:
        if log_dir is None:
            raise ValueError("log_dir must be provided when to_file is True")
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        log_path = log_dir / f"{run_name}_{stamp}.log"
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    # Third-party libraries are chatty at INFO and drown out pipeline messages.
    for noisy in ("matplotlib", "rasterio", "fiona", "numexpr", "shap", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True
    return log_path


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger.

    Parameters
    ----------
    name : str
        Usually ``__name__`` from the calling module.

    Returns
    -------
    logging.Logger
        A logger namespaced to the caller.
    """
    return logging.getLogger(name)
