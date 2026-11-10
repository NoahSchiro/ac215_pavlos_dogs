"""Per-user ridge-regression calibration."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def extreme_grid(n_side: int = 3) -> FloatArray:
    """Helper to get a grid of the looking points.

    So called 'extreme' because they are the edges, corners, and centers.
    """
    if n_side < 2:
        raise ValueError(f"n_side must be >= 2, got {n_side}")
    axis = np.linspace(0.0, 1.0, n_side)
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    return np.stack([xx.ravel(), yy.ravel()], axis=1)


def find_extreme_samples(
    labels: FloatArray, subjects: Sequence[str], n_side: int = 3
) -> dict[str, NDArray[np.intp]]:
    """Per subject, the indices of the samples nearest the grid anchors."""
    labels = np.asarray(labels, dtype=np.float64)
    if labels.ndim != 2 or labels.shape[1] != 2:
        raise ValueError(f"labels must have shape (N, 2), got {labels.shape}")
    if len(subjects) != len(labels):
        raise ValueError(f"got {len(subjects)} subject ids for {len(labels)} labels")
    grid = extreme_grid(n_side)

    selected: dict[str, NDArray[np.intp]] = {}
    for subject in sorted(set(subjects)):
        rows = np.flatnonzero(np.asarray(subjects) == subject).astype(np.intp)
        distances = np.linalg.norm(labels[rows][None, :, :] - grid[:, None, :], axis=-1)
        order = np.argsort(distances, axis=None, kind="stable")
        claimed_anchors: set[int] = set()
        claims: list[int] = []
        for flat in order:
            anchor, sample = divmod(int(flat), distances.shape[1])
            if anchor not in claimed_anchors and sample not in claims:
                claimed_anchors.add(anchor)
                claims.append(sample)
            if len(claimed_anchors) == len(grid):
                break
        selected[subject] = np.sort(rows[claims])
    return selected


@dataclass(frozen=True)
class RidgeFit:
    """Affine map `y = x @ weight + intercept` fitted by ridge regression."""

    weight: FloatArray
    """(D, D) slope matrix."""
    intercept: FloatArray
    """(D,) offset."""

    def apply(self, x: FloatArray) -> FloatArray:
        """Map (..., D) inputs through the correction."""
        return np.asarray(x, dtype=np.float64) @ self.weight + self.intercept


def fit_ridge(x: FloatArray, y: FloatArray, lamb: float = 0.1) -> RidgeFit:
    """Fit ridge regression of `y` on `x` with an unpenalised intercept."""
    if lamb < 0:
        raise ValueError(f"lamb must be >= 0, got {lamb}")
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or x.shape != y.shape:
        raise ValueError(
            f"x and y must be (N, D) with matching shapes, got {x.shape} and {y.shape}"
        )
    if len(x) == 0:
        raise ValueError("cannot fit ridge regression on zero samples")
    x_mean = x.mean(axis=0)
    y_mean = y.mean(axis=0)
    x_centered = x - x_mean
    weight = np.linalg.solve(
        x_centered.T @ x_centered + lamb * np.eye(x.shape[1]),
        x_centered.T @ (y - y_mean),
    )
    return RidgeFit(weight=weight, intercept=y_mean - x_mean @ weight)


def fit_user_calibration(
    predictions: FloatArray,
    targets: FloatArray,
    subjects: Sequence[str],
    selection_labels: FloatArray | None = None,
    lamb: float = 0.1,
    n_side: int = 3,
) -> dict[str, RidgeFit]:
    """One RidgeFit per subject, fitted on that subject's extreme samples."""
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    positions = targets if selection_labels is None else selection_labels
    chosen = find_extreme_samples(positions, subjects, n_side=n_side)
    return {
        subject: fit_ridge(predictions[rows], targets[rows], lamb=lamb)
        for subject, rows in chosen.items()
    }


def apply_user_calibration(
    predictions: FloatArray,
    subjects: Sequence[str],
    fits: dict[str, RidgeFit],
) -> FloatArray:
    """Apply each subject's correction; subjects without a fit pass through."""
    predictions = np.asarray(predictions, dtype=np.float64)
    if len(subjects) != len(predictions):
        raise ValueError(
            f"got {len(subjects)} subject ids for {len(predictions)} predictions"
        )
    calibrated = predictions.copy()
    for subject, fit in fits.items():
        mask = np.asarray([s == subject for s in subjects])
        if mask.any():
            calibrated[mask] = fit.apply(calibrated[mask])
    return calibrated
