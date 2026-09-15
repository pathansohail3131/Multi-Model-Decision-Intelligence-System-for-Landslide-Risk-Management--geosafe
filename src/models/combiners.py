"""Combination rules for the thematic models, and the disagreement measure.

Each rule takes the matrix of per-theme probabilities and returns one
susceptibility score. They are compared empirically in Stage 4 rather than
chosen by preference, and the comparison includes the full-feature model alone
— so that "no ensemble" is a candidate that can win.

A note on what winning means. Under a pure accuracy objective, if the
full-feature model has the best AUC-PR the ensemble should be discarded. Under
this project's decision objective it may still be preferred, because a single
model structurally cannot produce a disagreement signal: there is nothing for
it to disagree with. The ablation table therefore reports both the score and
whether a rule can produce confidence, and the choice is argued in the open in
``docs/methodology.md`` §5.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

__all__ = [
    "COMBINERS",
    "combine",
    "disagreement",
    "mean_combine",
    "weighted_mean_combine",
    "rank_average_combine",
    "noisy_or_combine",
    "max_combine",
]


def mean_combine(P: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Unweighted mean of the thematic probabilities.

    The natural baseline: it treats every physical theme as equally
    informative, which is a defensible prior when the themes were chosen for
    physical meaning rather than for measured strength.

    Parameters
    ----------
    P : numpy.ndarray, shape (n_samples, n_models)
        Per-model probabilities.
    weights : ignored

    Returns
    -------
    numpy.ndarray
        Combined score per sample.
    """
    return np.nanmean(P, axis=1)


def weighted_mean_combine(P: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Mean weighted by each model's measured skill.

    Parameters
    ----------
    P : numpy.ndarray, shape (n_samples, n_models)
        Per-model probabilities.
    weights : numpy.ndarray, optional
        Non-negative weights, typically each model's spatial-CV AUC-PR.
        Normalised internally. Falls back to the unweighted mean when absent.

    Returns
    -------
    numpy.ndarray
        Combined score per sample.

    Notes
    -----
    Weights are derived from out-of-fold performance only. Weighting by
    in-sample skill would hand the most weight to the most overfit member.
    """
    if weights is None:
        return mean_combine(P)
    w = np.asarray(weights, dtype="float64")
    w = np.clip(w, 0.0, None)
    if w.sum() <= 0:
        return mean_combine(P)
    w = w / w.sum()
    return np.nansum(P * w[None, :], axis=1)


def rank_average_combine(P: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Average of within-model percentile ranks.

    Scale-free, so it is immune to the calibration differences that would
    otherwise let a systematically confident model dominate a plain mean. The
    trade-off is that the output is a relative ordering rather than a
    probability, so it cannot be read as "the chance of failure here" and is
    not suitable for the final map without recalibration.

    Parameters
    ----------
    P : numpy.ndarray, shape (n_samples, n_models)
        Per-model probabilities.
    weights : numpy.ndarray, optional
        Optional model weights.

    Returns
    -------
    numpy.ndarray
        Combined score in [0, 1].
    """
    ranks = np.column_stack(
        [rankdata(P[:, j], method="average") / len(P) for j in range(P.shape[1])]
    )
    return weighted_mean_combine(ranks, weights)


def noisy_or_combine(P: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Probability that at least one theme indicates failure.

    ``1 - prod(1 - p_j)``. This encodes genuine hazard logic: a slope can fail
    for a purely geological reason *or* a purely hydrological one, and those
    are not averaged in reality — either suffices.

    Parameters
    ----------
    P : numpy.ndarray, shape (n_samples, n_models)
        Per-model probabilities.
    weights : ignored

    Returns
    -------
    numpy.ndarray
        Combined score per sample.

    Notes
    -----
    Noisy-OR assumes the themes are conditionally independent, which they are
    not — every theme partly reflects the same underlying terrain. It therefore
    tends to push scores up and is expected to over-flag. It is included
    because the physical argument is real and the size of the independence
    violation is worth measuring rather than assuming.
    """
    return 1.0 - np.prod(np.clip(1.0 - P, 0.0, 1.0), axis=1)


def max_combine(P: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Highest single-theme probability.

    The most conservative rule: one theme raising the alarm is enough. Useful
    as the upper bound on recall-oriented behaviour.
    """
    return np.nanmax(P, axis=1)


#: Registry of parameter-free combination rules, used by the Stage 4 ablation.
COMBINERS = {
    "thematic_mean": mean_combine,
    "thematic_weighted_mean": weighted_mean_combine,
    "thematic_rank_average": rank_average_combine,
    "thematic_noisy_or": noisy_or_combine,
    "thematic_max": max_combine,
}


def combine(
    name: str, P: np.ndarray, weights: np.ndarray | None = None
) -> np.ndarray:
    """Apply a named combination rule.

    Parameters
    ----------
    name : str
        Key from :data:`COMBINERS`.
    P : numpy.ndarray, shape (n_samples, n_models)
        Per-model probabilities.
    weights : numpy.ndarray, optional
        Model weights, used by rules that accept them.

    Returns
    -------
    numpy.ndarray
        Combined score per sample.
    """
    if name not in COMBINERS:
        raise ValueError(f"unknown combiner {name!r}; expected one of {sorted(COMBINERS)}")
    return COMBINERS[name](P, weights)


def disagreement(P: np.ndarray, method: str = "std") -> np.ndarray:
    """Measure how much the thematic models disagree, per sample.

    This is the raw material of the confidence raster, and therefore the actual
    product of the multi-model architecture. High susceptibility with high
    agreement means act; high susceptibility with high disagreement means send
    a survey team before committing resources.

    Parameters
    ----------
    P : numpy.ndarray, shape (n_samples, n_models)
        Per-model probabilities. **These must be calibrated.** Variance across
        uncalibrated models measures how differently they are *scaled*, not how
        differently they *judge*: a systematically confident model and a
        systematically conservative one will appear to disagree everywhere,
        including where they rank every pixel identically.
    method : {"std", "range", "entropy"}, default "std"
        ``"std"`` — standard deviation across models; the default, and directly
        interpretable as spread in probability units.
        ``"range"`` — max minus min; more sensitive to a single dissenting
        theme, which is often the physically interesting case.
        ``"entropy"`` — entropy of the normalised probability vector; responds
        to the shape of the disagreement rather than its magnitude.

    Returns
    -------
    numpy.ndarray
        Disagreement per sample. Higher means less agreement.
    """
    if method == "std":
        return np.nanstd(P, axis=1)
    if method == "range":
        return np.nanmax(P, axis=1) - np.nanmin(P, axis=1)
    if method == "entropy":
        Q = np.clip(P, 1e-9, 1.0)
        Q = Q / Q.sum(axis=1, keepdims=True)
        return -np.sum(Q * np.log(Q), axis=1)
    raise ValueError(f"unknown disagreement method {method!r}")


def confidence_from_disagreement(
    d: np.ndarray, max_possible: float | None = None
) -> np.ndarray:
    """Convert a disagreement measure into a confidence score in [0, 1].

    Parameters
    ----------
    d : numpy.ndarray
        Disagreement per sample.
    max_possible : float, optional
        Value mapped to zero confidence. Defaults to the observed maximum.

    Returns
    -------
    numpy.ndarray
        Confidence, where 1 means the themes agree completely.

    Notes
    -----
    Scaling by the observed maximum makes confidence *relative to this
    district*, not an absolute quantity. That is the honest reading: it ranks
    where a planner should be more or less sure, and it does not claim a
    calibrated probability of being correct.
    """
    d = np.asarray(d, dtype="float64")
    ceiling = float(max_possible if max_possible is not None else np.nanmax(d))
    if not np.isfinite(ceiling) or ceiling <= 0:
        return np.ones_like(d)
    return np.clip(1.0 - d / ceiling, 0.0, 1.0)
