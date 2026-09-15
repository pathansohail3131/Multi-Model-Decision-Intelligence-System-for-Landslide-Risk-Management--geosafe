"""Deterministic seeding across every random source the pipeline touches.

Reproducibility here is not a nicety. The headline claim of this project is
that a stacked ensemble scored 0.9225 against CatBoost's 0.9329 — a gap of
0.0104. Run-to-run variation from unseeded bootstrapping or unordered
thread reduction is easily that large, so without deterministic seeding the
ablation table in ``docs/results.md`` could not be defended at all.

What is and is not guaranteed
-----------------------------
Seeding the Python, NumPy and per-model RNGs makes a single-threaded run
bit-identical. Full determinism additionally requires bounded thread counts,
because floating-point addition is not associative and a parallel reduction
may sum partial results in a different order on each run. Calling
:func:`set_global_seed` with ``strict=True`` pins the thread-count environment
variables to 1, which makes runs reproducible at a real cost in speed. That
tradeoff is applied to the evaluation stages, where the numbers are the
deliverable, and relaxed for district-wide inference, where the output is a
smooth probability surface and a last-bit difference is immaterial.

The thread environment variables must be set before NumPy and the BLAS
libraries are imported to take effect, which is why the entry point calls this
module first.
"""

from __future__ import annotations

import logging
import os
import random

__all__ = ["set_global_seed", "seed_environment_report"]

logger = logging.getLogger(__name__)

#: Environment variables that bound thread pools in the numerical stack.
#: Set to "1" under strict mode to remove parallel-reduction nondeterminism.
_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def set_global_seed(seed: int, strict: bool = False) -> int:
    """Seed every global random source in the process.

    Parameters
    ----------
    seed : int
        The seed value. Propagated to :mod:`random`, NumPy's legacy global
        RNG, and ``PYTHONHASHSEED``.
    strict : bool, default False
        When True, also pin BLAS/OpenMP thread counts to 1 so that parallel
        floating-point reductions cannot reorder between runs. Use for any
        stage whose numeric output is reported; leave False for inference,
        where the speed matters more than the last bit.

    Returns
    -------
    int
        The seed that was applied, so callers can log it.

    Notes
    -----
    Individual estimators still need their own ``random_state``; this function
    does not reach inside scikit-learn, XGBoost or CatBoost. Model factories
    read the seed from the configuration and pass it explicitly, which is
    deliberate — relying on a global RNG for model seeding is precisely how
    "reproducible" pipelines quietly stop being reproducible.
    """
    # PYTHONHASHSEED only affects interpreters started after it is set, so this
    # helps subprocesses rather than the current one. Set anyway for children.
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover - numpy is a hard dependency
        logger.warning("NumPy unavailable; skipped NumPy seeding")

    if strict:
        for var in _THREAD_ENV_VARS:
            os.environ[var] = "1"
        logger.info("Strict determinism on: thread pools pinned to 1")

    logger.info("Global random seed set to %d (strict=%s)", seed, strict)
    return seed


def seed_environment_report() -> dict[str, str]:
    """Capture the seeding-relevant environment for run provenance.

    The returned mapping is written into each stage's report so that a result
    can be traced back to the exact determinism settings that produced it.

    Returns
    -------
    dict of str to str
        Environment variable names mapped to their current values, with
        ``"<unset>"`` for variables that are not set.
    """
    tracked = ("PYTHONHASHSEED",) + _THREAD_ENV_VARS
    return {var: os.environ.get(var, "<unset>") for var in tracked}
