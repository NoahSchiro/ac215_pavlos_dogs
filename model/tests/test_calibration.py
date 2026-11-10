"""Tests for per-user ridge calibration and extreme-sample selection."""

from typing import Any

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from src.training.calibration import (
    RidgeFit,
    apply_user_calibration,
    extreme_grid,
    find_extreme_samples,
    fit_ridge,
    fit_user_calibration,
)
from src.training.loop import collect_predictions, evaluate
from src.training.model import GazeNet


def test_extreme_grid_is_the_nine_point_lattice() -> None:
    grid = extreme_grid()
    assert grid.shape == (9, 2)
    assert set(map(tuple, grid)) == {
        (x, y) for x in (0.0, 0.5, 1.0) for y in (0.0, 0.5, 1.0)
    }


def test_extreme_grid_rejects_a_degenerate_lattice() -> None:
    with pytest.raises(ValueError, match="n_side"):
        extreme_grid(1)


def test_find_extreme_samples_selects_the_anchor_points() -> None:
    """Anchor-placed samples beat interior filler, wherever they sit."""
    rng = np.random.default_rng(0)
    grid = extreme_grid()
    labels = np.vstack([grid, rng.uniform(0.25, 0.75, (20, 2))])
    order = rng.permutation(len(labels))
    chosen = find_extreme_samples(labels[order], ["p00"] * len(labels))
    picked = labels[order][chosen["p00"]]
    assert sorted(map(tuple, np.round(picked, 6))) == sorted(map(tuple, grid))


def test_find_extreme_samples_is_per_subject() -> None:
    rng = np.random.default_rng(1)
    grid = extreme_grid()
    p0 = grid + rng.uniform(-0.02, 0.02, (9, 2))
    p1 = grid + rng.uniform(-0.02, 0.02, (9, 2))
    filler = rng.uniform(0.3, 0.7, (10, 2))
    labels = np.vstack([p0, p1, filler])
    subjects = ["p00"] * 9 + ["p01"] * 9 + ["p00"] * 5 + ["p01"] * 5
    chosen = find_extreme_samples(labels, subjects)
    assert len(chosen) == 2
    assert np.array_equal(chosen["p00"], np.arange(9))
    assert np.array_equal(chosen["p01"], np.arange(9, 18))


def test_find_extreme_samples_tolerates_fewer_samples_than_anchors() -> None:
    labels = np.array([[0.0, 0.0], [0.9, 0.1], [0.5, 0.5], [1.0, 1.0]])
    chosen = find_extreme_samples(labels, ["p00"] * 4)
    assert chosen["p00"].tolist() == [0, 1, 2, 3]


def test_find_extreme_samples_rejects_mismatched_subject_ids() -> None:
    with pytest.raises(ValueError, match="subject ids"):
        find_extreme_samples(np.zeros((3, 2)), ["p00", "p00"])


def test_find_extreme_samples_rejects_non_planar_labels() -> None:
    with pytest.raises(ValueError, match=r"\(N, 2\)"):
        find_extreme_samples(np.zeros((3, 3)), ["p00"] * 3)


def test_fit_ridge_recovers_a_known_affine_map() -> None:
    rng = np.random.default_rng(2)
    x = rng.uniform(0.0, 1.0, (200, 2))
    weight = np.array([[0.8, -0.2], [0.1, 0.9]])
    intercept = np.array([0.05, -0.1])
    y = x @ weight + intercept
    fit = fit_ridge(x, y, lamb=1e-9)
    assert np.allclose(fit.weight, weight, atol=1e-8)
    assert np.allclose(fit.intercept, intercept, atol=1e-8)
    assert np.allclose(fit.apply(x), y)


def test_fit_ridge_shrinks_toward_the_mean_with_large_lambda() -> None:
    rng = np.random.default_rng(3)
    x = rng.uniform(0.0, 1.0, (50, 2))
    y = x + 0.1
    fit = fit_ridge(x, y, lamb=1e12)
    assert np.allclose(fit.weight, 0.0, atol=1e-6)
    # Only the mean offset survives: everything maps to mean(y).
    assert np.allclose(fit.apply(x), y.mean(axis=0))


def test_fit_ridge_with_one_sample_corrects_only_the_offset() -> None:
    x = np.array([[0.2, 0.3]])
    y = np.array([[0.5, 0.7]])
    fit = fit_ridge(x, y)
    assert np.allclose(fit.weight, 0.0)
    assert np.allclose(fit.apply(x), y)


def test_fit_ridge_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="matching shapes"):
        fit_ridge(np.zeros((3, 2)), np.zeros((3, 3)))
    with pytest.raises(ValueError, match="zero samples"):
        fit_ridge(np.zeros((0, 2)), np.zeros((0, 2)))
    with pytest.raises(ValueError, match="lamb"):
        fit_ridge(np.zeros((3, 2)), np.zeros((3, 2)), lamb=-1.0)


def test_apply_user_calibration_passes_unknown_subjects_through() -> None:
    predictions = np.array([[0.1, 0.1], [0.5, 0.5]])
    fit = RidgeFit(weight=np.zeros((2, 2)), intercept=np.array([0.4, 0.6]))
    calibrated = apply_user_calibration(predictions, ["p00", "p01"], {"p00": fit})
    assert calibrated[0].tolist() == pytest.approx([0.4, 0.6])
    assert calibrated[1].tolist() == pytest.approx([0.5, 0.5])


def test_apply_user_calibration_rejects_mismatched_subject_ids() -> None:
    fit = RidgeFit(weight=np.zeros((2, 2)), intercept=np.zeros(2))
    with pytest.raises(ValueError, match="subject ids"):
        apply_user_calibration(np.zeros((2, 2)), ["p00"], {"p00": fit})


class GridGaze(Dataset[dict[str, Any]]):
    """Per subject: samples near the nine grid anchors plus interior filler."""

    def __init__(self, subjects: tuple[int, ...], per_subject: int = 12) -> None:
        if per_subject < 9:
            raise ValueError("per_subject must leave room for the nine anchors")
        rng = np.random.default_rng(0)
        anchors = extreme_grid()
        self.items: list[dict[str, Any]] = []
        for subject in subjects:
            labels = np.vstack(
                [
                    anchors + rng.uniform(-0.01, 0.01, (9, 2)),
                    rng.uniform(0.25, 0.75, (per_subject - 9, 2)),
                ]
            )
            for i, label in enumerate(labels):
                self.items.append(
                    {
                        "sample_id": f"mpiifacegaze/p{subject:02d}/day01/{i:04d}",
                        "face": rng.integers(0, 255, (224, 224, 3), dtype=np.uint8),
                        "left_eye": rng.integers(0, 255, (36, 60, 3), dtype=np.uint8),
                        "right_eye": rng.integers(0, 255, (36, 60, 3), dtype=np.uint8),
                        "label": label.astype(np.float32),
                        "direction": np.array([0.0, 0.0, 1.0], dtype=np.float32),
                        "R_norm": np.eye(3),
                    }
                )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.items[idx]


def test_evaluate_applies_a_fitted_calibration() -> None:
    """main fits the correction, then hands it to evaluate for scoring."""
    torch.manual_seed(0)
    model = GazeNet(pretrained=False, use_eyes=False)
    device = torch.device("cpu")
    loader = DataLoader(GridGaze((0, 1)), batch_size=4, num_workers=0)
    raw, loss = evaluate(model, loader, device)
    assert loss >= 0.0

    predictions, targets, screen_labels, subjects = collect_predictions(
        model, loader, device
    )
    assert predictions.shape == (24, 2)
    assert targets.shape == (24, 2)
    assert screen_labels.shape == (24, 2)
    assert len(subjects) == 24

    fits = fit_user_calibration(
        predictions.numpy(),
        targets.numpy(),
        subjects,
        selection_labels=screen_labels.numpy(),
    )
    calibrated, _ = evaluate(model, loader, device, calibration=fits)
    assert calibrated.count == raw.count == 24
    assert set(calibrated.per_participant) == {"p00", "p01"}
    # A fitted correction changes the scores; an empty one must not.
    assert calibrated.mean != raw.mean
    uncalibrated, _ = evaluate(model, loader, device, calibration={})
    assert uncalibrated.mean == raw.mean
    assert uncalibrated.median == raw.median


def test_evaluate_applies_calibration_in_direction_mode() -> None:
    """Direction models are sampled at 2D extremes but corrected in 3D."""
    torch.manual_seed(0)
    model = GazeNet(output_mode="gaze_direction", pretrained=False, use_eyes=False)
    device = torch.device("cpu")
    loader = DataLoader(GridGaze((0,)), batch_size=4, num_workers=0)
    raw, _ = evaluate(model, loader, device)

    predictions, targets, screen_labels, subjects = collect_predictions(
        model, loader, device
    )
    assert predictions.shape == (12, 3)
    assert targets.shape == (12, 3)
    assert screen_labels.shape == (12, 2)  # screen labels, not directions
    fits = fit_user_calibration(
        predictions.numpy(),
        targets.numpy(),
        subjects,
        selection_labels=screen_labels.numpy(),
    )
    calibrated, _ = evaluate(model, loader, device, calibration=fits)
    # Every direction label is (0, 0, 1), so the per-user fit maps every
    # prediction to exactly that vector and the angular error vanishes.
    assert calibrated.mean == pytest.approx(0.0, abs=1e-5)
    assert raw.mean > calibrated.mean


def test_collect_predictions_falls_back_to_targets_without_labels() -> None:
    torch.manual_seed(0)
    model = GazeNet(output_mode="gaze_direction", pretrained=False, use_eyes=False)
    dataset = GridGaze((0,))
    for item in dataset.items:
        item.pop("label")
    loader = DataLoader(dataset, batch_size=6, num_workers=0)
    predictions, targets, screen_labels, subjects = collect_predictions(
        model, loader, torch.device("cpu")
    )
    assert predictions.shape == (12, 3)
    assert torch.equal(screen_labels, targets)
    assert len(subjects) == 12


def test_collect_predictions_rejects_an_empty_loader() -> None:
    class Empty(Dataset[dict[str, Any]]):
        def __len__(self) -> int:
            return 0

        def __getitem__(self, idx: int) -> dict[str, Any]:
            raise IndexError(idx)

    with pytest.raises(ValueError, match="no batches"):
        collect_predictions(
            GazeNet(pretrained=False, use_eyes=False),
            DataLoader(Empty(), batch_size=2),
            torch.device("cpu"),
        )
