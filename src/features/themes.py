"""Thematic feature subsets — the mechanism that makes the ensemble diverse.

The previous architecture induced diversity by varying the *algorithm* while
every learner saw identical columns. Three of the five were tree ensembles, so
their errors were correlated, and the stack lost to its own best member
(0.9225 against 0.9329).

Here diversity is induced by restricting what each model can *see*. Two models
sharing no input columns cannot make correlated errors through a shared input,
which is a structural guarantee rather than an assumption. The partition is
also physically meaningful, which is what makes the disagreement signal usable:
"flagged by terrain but not by geology" names a specific kind of uncertainty a
survey team can be sent to resolve.

One honest caveat, recorded rather than engineered around: ``dist_fault`` is a
DEM-derived structural-lineament proxy, not mapped fault data, so the
geological and topographic themes share no *column* but do share an *upstream
source*. Stage 4 measures the resulting prediction correlation directly rather
than assuming it is small.
"""

from __future__ import annotations

import pandas as pd

from src.utils.logging_setup import get_logger

__all__ = [
    "theme_features",
    "all_theme_features",
    "subset_to_theme",
    "validate_partition",
    "BENCHMARK_NAME",
]

logger = get_logger(__name__)

#: Name used for the full-feature benchmark model. It is not a thematic member
#: and never contributes to the disagreement/confidence calculation.
BENCHMARK_NAME = "full"


def theme_features(cfg, theme: str, available: list[str] | None = None) -> list[str]:
    """Return the encoded feature names belonging to one theme.

    Parameters
    ----------
    cfg : Config
        Resolved project configuration.
    theme : str
        Theme name, or ``"full"`` for the benchmark model.
    available : list of str, optional
        Feature names actually present after encoding. When given, the result
        is intersected with it and any absence is logged. This matters because
        the encoder emits ``is_flat`` only when a slope column is present, so
        the topographic theme's width is data-dependent.

    Returns
    -------
    list of str
        Feature names for the theme, in configuration order.
    """
    if theme == BENCHMARK_NAME:
        if available is None:
            raise ValueError(
                "the benchmark model needs the encoded feature list; "
                "pass available=<encoded columns>"
            )
        return list(available)

    wanted = cfg.theme_features(theme)
    if available is None:
        return wanted

    present = [f for f in wanted if f in available]
    missing = [f for f in wanted if f not in available]
    if missing:
        logger.warning(
            "theme %r: %d configured feature(s) absent after encoding: %s",
            theme,
            len(missing),
            missing,
        )
    return present


def all_theme_features(cfg, available: list[str]) -> dict[str, list[str]]:
    """Return the feature list for every thematic model plus the benchmark.

    Parameters
    ----------
    cfg : Config
        Resolved project configuration.
    available : list of str
        Encoded feature names.

    Returns
    -------
    dict
        Model name mapped to its feature list. Thematic models first, then
        ``"full"``.
    """
    out = {
        theme: theme_features(cfg, theme, available) for theme in cfg.theme_names
    }
    out[BENCHMARK_NAME] = list(available)
    return out


def subset_to_theme(
    X: pd.DataFrame, cfg, theme: str
) -> pd.DataFrame:
    """Slice an encoded feature frame down to one theme.

    Parameters
    ----------
    X : pandas.DataFrame
        Encoded features.
    cfg : Config
        Resolved project configuration.
    theme : str
        Theme name, or ``"full"``.

    Returns
    -------
    pandas.DataFrame
        The theme's columns, in configuration order.
    """
    columns = theme_features(cfg, theme, available=list(X.columns))
    return X[columns]


def validate_partition(cfg) -> dict:
    """Check that the themes tile the conditioning factors exactly once each.

    Overlap would reintroduce precisely the shared input that made the previous
    stack's members agree for the wrong reason: agreement between two models
    that both see slope measures slope, not independent evidence. A gap would
    mean a factor is silently unused.

    Parameters
    ----------
    cfg : Config
        Resolved project configuration.

    Returns
    -------
    dict
        Report with ``is_partition``, any ``overlaps`` and any ``unassigned``.
    """
    assigned: dict[str, list[str]] = {}
    for theme in cfg.theme_names:
        for factor in cfg.get("themes", theme, "factors"):
            assigned.setdefault(factor, []).append(theme)

    overlaps = {f: t for f, t in assigned.items() if len(t) > 1}
    configured = set(cfg.raster_factors)
    unassigned = sorted(configured - set(assigned))
    unknown = sorted(set(assigned) - configured)

    report = {
        "n_factors": len(configured),
        "n_assigned": len(assigned),
        "themes": {
            theme: list(cfg.get("themes", theme, "factors")) for theme in cfg.theme_names
        },
        "overlaps": overlaps,
        "unassigned": unassigned,
        "unknown": unknown,
        "is_partition": not overlaps and not unassigned and not unknown,
    }

    if report["is_partition"]:
        logger.info(
            "Theme partition valid: %d factors across %d themes, no overlap",
            len(configured),
            len(cfg.theme_names),
        )
    else:
        logger.error("Theme partition INVALID: %s", report)
    return report
