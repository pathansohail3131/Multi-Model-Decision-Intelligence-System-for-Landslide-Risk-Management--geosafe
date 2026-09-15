"""Spatial block cross-validation.

Landslide points cluster: 86 of 417 positives sit within 100 m of another
positive, with a minimum separation of 0.65 m. Under random k-fold, a test
point can therefore sit metres from a training point on the same slope, and the
model is scored on its ability to recall a neighbour rather than to generalise
to unseen terrain. The score that produces is not wrong arithmetic — it is a
correct answer to the wrong question.

Spatial block CV answers the right one: tile the district into square blocks,
assign whole blocks to folds, and score each fold on terrain the model has
never seen.

Block size
----------
Stage 1 measures the empirical variogram of the target and finds it is already
at 87% of the sill by the 500 m lag and effectively at the sill from ~2.5 km.
That short range is partly real and partly an artefact of the sampling design —
negatives were drawn with a ≥100 m exclusion buffer and spread across the
district — which makes a single variogram-derived number a weak basis for
choosing one block size.

So this module does not pick one. :func:`block_size_sensitivity` sweeps a range
of block sizes and reports the spatial-versus-random gap at each. The shape of
that curve is a far more honest answer than any single figure, and it is what
``docs/results.md`` reports.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedKFold

from src.utils.logging_setup import get_logger

__all__ = [
    "assign_blocks",
    "SpatialBlockCV",
    "random_cv_splits",
    "summarise_split",
]

logger = get_logger(__name__)


def assign_blocks(
    xs: np.ndarray, ys: np.ndarray, block_size_m: float
) -> np.ndarray:
    """Assign each point to a square spatial block.

    Parameters
    ----------
    xs, ys : numpy.ndarray
        Projected coordinates in metres.
    block_size_m : float
        Block side length in metres.

    Returns
    -------
    numpy.ndarray
        Integer block id per point. Ids are dense (0..n_blocks-1) so they can
        index arrays directly.

    Notes
    -----
    Blocks are anchored at the minimum coordinate rather than at the CRS
    origin, so that the tiling covers the district compactly and the block
    count does not depend on how far the district sits from the UTM origin.
    """
    xs = np.asarray(xs, dtype="float64")
    ys = np.asarray(ys, dtype="float64")
    col = np.floor((xs - xs.min()) / block_size_m).astype("int64")
    row = np.floor((ys - ys.min()) / block_size_m).astype("int64")
    # Combine the 2-D block index into a single key, then densify.
    key = row * (col.max() + 1) + col
    _, dense = np.unique(key, return_inverse=True)
    return dense


class SpatialBlockCV:
    """Cross-validator that keeps whole spatial blocks within a fold.

    Blocks are assigned to folds greedily by positive count, which keeps the
    positive class spread across folds. With 417 positives over 5 folds, a
    purely random block assignment can easily leave a fold with very few
    positives, and AUC-PR on such a fold is close to meaningless.

    Parameters
    ----------
    n_splits : int, default 5
        Number of folds.
    block_size_m : float, default 5000.0
        Block side length in metres.
    random_state : int, default 42
        Seed for the shuffle applied before greedy assignment.

    Notes
    -----
    Adjacent blocks land in different folds, so a test point at a block edge
    can still be near a training point across the boundary. That residual
    proximity is why the block-size sweep matters: larger blocks mean a
    smaller share of test points near a fold boundary. The effect is reported
    rather than eliminated, since eliminating it entirely would require
    discarding buffer zones and shrinking an already small positive class.
    """

    def __init__(
        self,
        n_splits: int = 5,
        block_size_m: float = 5000.0,
        random_state: int = 42,
    ) -> None:
        self.n_splits = n_splits
        self.block_size_m = block_size_m
        self.random_state = random_state

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Return the number of folds."""
        return self.n_splits

    def split(self, X, y, coords: np.ndarray):
        """Generate train/test index arrays.

        Parameters
        ----------
        X : array-like
            Ignored; present for API compatibility.
        y : array-like
            Binary target, used to balance positives across folds.
        coords : numpy.ndarray, shape (n, 2)
            Projected coordinates in metres.

        Yields
        ------
        train_idx, test_idx : numpy.ndarray
            Index arrays for each fold.
        """
        y = np.asarray(y).astype(int)
        blocks = assign_blocks(coords[:, 0], coords[:, 1], self.block_size_m)
        unique_blocks = np.unique(blocks)

        # Positives per block drives the greedy balancing.
        pos_per_block = np.array(
            [int(y[blocks == b].sum()) for b in unique_blocks], dtype="int64"
        )
        size_per_block = np.array(
            [int((blocks == b).sum()) for b in unique_blocks], dtype="int64"
        )

        rng = np.random.default_rng(self.random_state)
        order = rng.permutation(len(unique_blocks))
        # Descending positive count, ties broken by the seeded shuffle, so the
        # assignment is deterministic but not spatially systematic.
        order = order[np.argsort(-pos_per_block[order], kind="stable")]

        fold_pos = np.zeros(self.n_splits, dtype="int64")
        fold_size = np.zeros(self.n_splits, dtype="int64")
        block_fold = np.empty(len(unique_blocks), dtype="int64")

        for i in order:
            # Send each block to the fold with fewest positives so far,
            # breaking ties on total size to keep folds comparable.
            target = int(np.lexsort((fold_size, fold_pos))[0])
            block_fold[i] = target
            fold_pos[target] += pos_per_block[i]
            fold_size[target] += size_per_block[i]

        point_fold = block_fold[blocks]
        logger.debug(
            "Spatial blocks: %d blocks at %.0f m -> folds with positives %s",
            len(unique_blocks),
            self.block_size_m,
            fold_pos.tolist(),
        )

        indices = np.arange(len(y))
        for fold in range(self.n_splits):
            test_mask = point_fold == fold
            if not test_mask.any():
                logger.warning("Spatial fold %d is empty; skipping", fold)
                continue
            yield indices[~test_mask], indices[test_mask]


def random_cv_splits(y: np.ndarray, n_splits: int = 5, random_state: int = 42):
    """Generate stratified random k-fold splits.

    Used **only** to quantify the optimism that ignoring spatial structure
    injects. It is never used for model selection.

    Parameters
    ----------
    y : numpy.ndarray
        Binary target.
    n_splits : int, default 5
        Number of folds.
    random_state : int, default 42
        Seed.

    Yields
    ------
    train_idx, test_idx : numpy.ndarray
    """
    splitter = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    yield from splitter.split(np.zeros(len(y)), y)


def summarise_split(
    y: np.ndarray, coords: np.ndarray, block_size_m: float, n_splits: int, seed: int
) -> dict:
    """Describe a spatial blocking scheme without running any model.

    Reports the block count, per-fold positive counts, and the distribution of
    distances from each test point to its nearest training point — the last
    being the direct measure of how much proximity leakage the scheme still
    permits.

    Parameters
    ----------
    y : numpy.ndarray
        Binary target.
    coords : numpy.ndarray, shape (n, 2)
        Projected coordinates in metres.
    block_size_m : float
        Block side length.
    n_splits : int
        Number of folds.
    seed : int
        Seed for fold assignment.

    Returns
    -------
    dict
        Diagnostics for the scheme.
    """
    from scipy.spatial import cKDTree

    cv = SpatialBlockCV(n_splits=n_splits, block_size_m=block_size_m, random_state=seed)
    blocks = assign_blocks(coords[:, 0], coords[:, 1], block_size_m)

    fold_positives, nearest = [], []
    for train_idx, test_idx in cv.split(None, y, coords):
        fold_positives.append(int(np.asarray(y)[test_idx].sum()))
        tree = cKDTree(coords[train_idx])
        nearest.append(tree.query(coords[test_idx])[0])

    nearest_all = np.concatenate(nearest) if nearest else np.array([0.0])
    return {
        "block_size_m": float(block_size_m),
        "n_blocks": int(len(np.unique(blocks))),
        "mean_points_per_block": round(float(len(y) / len(np.unique(blocks))), 2),
        "positives_per_fold": fold_positives,
        "min_positives_in_a_fold": int(min(fold_positives)) if fold_positives else 0,
        "test_to_train_distance": {
            "median_m": round(float(np.median(nearest_all)), 1),
            "mean_m": round(float(nearest_all.mean()), 1),
            "pct_within_100m": round(float((nearest_all < 100).mean()), 4),
            "pct_within_500m": round(float((nearest_all < 500).mean()), 4),
            "pct_within_1000m": round(float((nearest_all < 1000).mean()), 4),
        },
    }
