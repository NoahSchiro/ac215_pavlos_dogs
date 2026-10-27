"""Tests for metrics, the model and the training loop.

Uses a synthetic in-memory dataset, so nothing here touches the data pipeline.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime
import pytest
import torch
import wandb
from torch.utils.data import DataLoader, Dataset

from src.data.utils import prepare_images
from src.training.config import TrainConfig
from src.training.export import export_onnx, load_checkpoint, save_checkpoint
from src.training.logging import ConsoleLogger, WandbLogger
from src.training.loop import (
    evaluate,
    fit,
)
from src.training.metrics import (
    angular_error_deg,
    cm_to_degrees,
    fraction_to_cm,
    screen_fraction_error,
    summarize,
)
from src.training.model import GazeNet


class FakeGaze(Dataset[dict[str, Any]]):
    """Stand-in with the same keys and dtypes a real Dataset emits."""

    def __init__(self, subjects: tuple[int, ...], per_subject: int = 4) -> None:
        self.items: list[dict[str, Any]] = []
        rng = np.random.default_rng(0)
        for subject in subjects:
            for i in range(per_subject):
                self.items.append(
                    {
                        "sample_id": f"mpiifacegaze/p{subject:02d}/day01/{i:04d}",
                        "face": rng.integers(0, 255, (224, 224, 3), dtype=np.uint8),
                        "left_eye": rng.integers(0, 255, (36, 60, 3), dtype=np.uint8),
                        "right_eye": rng.integers(0, 255, (36, 60, 3), dtype=np.uint8),
                        "label": np.array([0.4, 0.6], dtype=np.float32),
                        "direction": np.array([0.0, 0.0, 1.0], dtype=np.float32),
                        "R_norm": np.eye(3),
                    }
                )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.items[idx]


def test_screen_fraction_error_is_euclidean() -> None:
    predicted = torch.tensor([[0.0, 0.0], [0.5, 0.5]])
    target = torch.tensor([[3.0, 4.0], [0.5, 0.5]])
    assert screen_fraction_error(predicted, target).tolist() == [5.0, 0.0]


def test_screen_fraction_error_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        screen_fraction_error(torch.zeros(2, 2), torch.zeros(3, 2))


def test_angular_error_on_known_angles() -> None:
    predicted = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    target = torch.tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    assert angular_error_deg(predicted, target).tolist() == pytest.approx(
        [0.0, 90.0], abs=1e-4
    )


def test_fraction_converts_to_cm_then_degrees() -> None:
    # 286x179mm screen -> 33.8cm diagonal; 10% error -> 3.4cm -> ~3.2deg at 60cm
    cm = fraction_to_cm(0.1, 286.0, 179.0)
    assert cm == pytest.approx(3.37, abs=0.02)
    assert cm_to_degrees(cm, 60.0) == pytest.approx(3.22, abs=0.05)


def test_conversions_reject_nonsense_geometry() -> None:
    with pytest.raises(ValueError, match="screen dimensions"):
        fraction_to_cm(0.1, 0.0, 179.0)
    with pytest.raises(ValueError, match="viewing distance"):
        cm_to_degrees(3.0, 0.0)


def test_summarize_nests_stats_per_participant() -> None:
    stats = summarize(torch.tensor([0.1, 0.3, 0.2, 0.4]), ["p00", "p00", "p01", "p01"])
    assert stats.count == 4
    assert stats.mean == pytest.approx(0.25)
    assert stats.per_participant["p00"].mean == pytest.approx(0.2)
    assert stats.per_participant["p01"].count == 2

    worst, worst_stats = stats.worst_participant()  # type: ignore[misc]
    assert worst == "p01"
    assert worst_stats.mean == pytest.approx(0.3)


def test_summarize_rejects_mismatched_subject_ids() -> None:
    with pytest.raises(ValueError, match="subject ids"):
        summarize(torch.tensor([0.1, 0.2]), ["p00"])


def test_summarize_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="empty"):
        summarize(torch.tensor([]))


@pytest.mark.parametrize("mode,dim", [("screen_fraction", 2), ("gaze_direction", 3)])
def test_output_shape_and_range_per_mode(mode: str, dim: int) -> None:
    model = GazeNet(output_mode=mode, pretrained=False)  # type: ignore[arg-type]
    out = model(
        torch.zeros(2, 3, 224, 224),
        torch.zeros(2, 3, 36, 60),
        torch.zeros(2, 3, 36, 60),
    )
    assert out.shape == (2, dim)
    if mode == "screen_fraction":
        assert torch.all((out >= 0) & (out <= 1))
    else:
        assert torch.linalg.vector_norm(out, dim=-1).tolist() == pytest.approx(
            [1.0, 1.0]
        )


def test_eye_tower_is_shared_across_both_eyes() -> None:
    with_eyes = GazeNet(pretrained=False, use_eyes=True)
    without = GazeNet(pretrained=False, use_eyes=False)
    # One shared tower, not two: the delta stays well under 0.4M parameters.
    assert with_eyes.parameter_count() - without.parameter_count() < 400_000


def test_ablating_eyes_requires_only_the_face() -> None:
    model = GazeNet(pretrained=False, use_eyes=False)
    assert model(torch.zeros(2, 3, 224, 224)).shape == (2, 2)


def test_eye_crops_are_required_when_enabled() -> None:
    model = GazeNet(pretrained=False, use_eyes=True)
    with pytest.raises(ValueError, match="requires both eye crops"):
        model(torch.zeros(2, 3, 224, 224))


def test_rejects_unknown_output_mode() -> None:
    with pytest.raises(ValueError, match="output_mode"):
        GazeNet(output_mode="sideways")  # type: ignore[arg-type]


def test_rejects_nonsense_dropout() -> None:
    with pytest.raises(ValueError, match="dropout"):
        GazeNet(pretrained=False, dropout=1.5)


def test_dropout_drops_in_train_and_is_off_in_eval() -> None:
    model = GazeNet(pretrained=False, dropout=0.5)
    face, eyes = torch.zeros(2, 3, 224, 224), torch.zeros(2, 3, 36, 60)
    model.eval()
    assert torch.equal(model(face, eyes, eyes), model(face, eyes, eyes))
    model.train()
    assert not torch.equal(model(face, eyes, eyes), model(face, eyes, eyes))


def test_output_mode_is_keyword_only() -> None:
    with pytest.raises(TypeError):
        GazeNet("screen_fraction")  # type: ignore[misc]


def test_prepare_images_applies_imagenet_statistics() -> None:
    batch = torch.full((2, 36, 60, 3), 255, dtype=torch.uint8)
    out = prepare_images(batch, torch.device("cpu"), imagenet=True)
    assert out.shape == (2, 3, 36, 60)
    assert out.max().item() == pytest.approx((1.0 - 0.406) / 0.225, abs=1e-4)


def test_prepare_images_can_skip_imagenet_statistics() -> None:
    batch = torch.full((2, 36, 60, 3), 255, dtype=torch.uint8)
    out = prepare_images(batch, torch.device("cpu"), imagenet=False)
    assert out.max().item() == pytest.approx(1.0)


def test_prepare_images_rejects_wrong_layout() -> None:
    with pytest.raises(ValueError, match="B, H, W, 3"):
        prepare_images(
            torch.zeros(2, 3, 36, 60, dtype=torch.uint8), torch.device("cpu")
        )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Parse a ConsoleLogger JSON-lines file back into records."""
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_console_logger_appends_json_lines(tmp_path: Path) -> None:
    logger = ConsoleLogger(tmp_path / "run.jsonl")
    logger.log({"epoch": 0, "val_mean": 0.5})
    logger.log({"epoch": 1, "val_mean": 0.4})
    assert [record["val_mean"] for record in read_jsonl(tmp_path / "run.jsonl")] == [
        0.5,
        0.4,
    ]


class FakeWandbSummary:
    """Mirrors wandb.Run.summary: an attribute with an update method."""

    def __init__(self) -> None:
        self.updates: list[dict[str, Any]] = []

    def update(self, data: dict[str, Any]) -> None:
        self.updates.append(data)


class FakeWandbRun:
    """Records what WandbLogger does, standing in for wandb.Run."""

    def __init__(self) -> None:
        self.init_kwargs: dict[str, Any] = {}
        self.logged: list[dict[str, Any]] = []
        self.defined_metrics: list[tuple[str, dict[str, Any]]] = []
        self.steps: list[int | None] = []
        self.summary = FakeWandbSummary()
        self.finished = False

    def record_init(self, **kwargs: Any) -> FakeWandbRun:
        self.init_kwargs = kwargs
        return self

    def define_metric(self, name: str, **kwargs: Any) -> None:
        self.defined_metrics.append((name, kwargs))

    def log(self, data: dict[str, Any], step: int | None = None) -> None:
        self.logged.append(data)
        self.steps.append(step)

    def finish(self) -> None:
        self.finished = True


def test_wandb_logger_tracks_config_pins_step_and_finishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeWandbRun()
    monkeypatch.setattr(wandb, "init", fake.record_init)

    logger = WandbLogger(
        "gaze", {"epochs": 2, "checkpoint_dir": Path("checkpoints/run")}
    )
    logger.log({"epoch": 0, "train_loss": 1.0})
    logger.log({"epoch": 1, "train_loss": 0.5})
    logger.summarize({"test_mean": 0.1})
    logger.finish()

    assert fake.init_kwargs["project"] == "gaze"
    # Paths are coerced so the config always serializes.
    assert fake.init_kwargs["config"] == {
        "epochs": 2,
        "checkpoint_dir": "checkpoints/run",
    }
    # Steps are pinned to the epoch, so a resumed run continues its curves.
    assert fake.steps == [0, 1]
    assert fake.summary.updates == [{"test_mean": 0.1}]
    assert fake.finished


def test_checkpoint_round_trips_weights_and_optimizer(tmp_path: Path) -> None:
    model = GazeNet(pretrained=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = tmp_path / "last.pt"
    save_checkpoint(path, model, optimizer, 4, 0.25, TrainConfig())

    restored = GazeNet(pretrained=False)
    assert load_checkpoint(path, restored) == (4, 0.25)
    for a, b in zip(
        model.state_dict().values(), restored.state_dict().values(), strict=True
    ):
        assert torch.equal(a, b)


def test_fit_decays_kernels_and_exempts_norms_and_biases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The optimizer grouping inside `fit`: 1-D params are weight-decay exempt."""
    real_adamw = torch.optim.AdamW
    captured: list[list[dict[str, Any]]] = []

    def spy(params: Any, **kwargs: Any) -> torch.optim.Optimizer:
        captured.append(params)
        return real_adamw(params, **kwargs)

    monkeypatch.setattr(torch.optim, "AdamW", spy)
    model = GazeNet(pretrained=False)
    fit(
        model,
        DataLoader(FakeGaze((0,)), batch_size=2),
        DataLoader(FakeGaze((1,)), batch_size=2),
        torch.device("cpu"),
        TrainConfig(epochs=1, batch_size=2, num_workers=0),
        tmp_path,
        ConsoleLogger(tmp_path / "run.jsonl"),
    )

    decay, no_decay = captured[0]
    # An exact partition: every parameter lands in exactly one group.
    assert {
        id(p) for group in captured[0] for p in group["params"]
    } == {id(p) for p in model.parameters()}
    # Norm scales and biases are 1-D; conv and linear kernels are 2-D+.
    assert all(p.ndim >= 2 for p in decay["params"])
    assert decay["weight_decay"] == 1e-4
    assert all(p.ndim <= 1 for p in no_decay["params"])
    assert no_decay["weight_decay"] == 0.0


def test_fit_runs_with_every_augmentation(tmp_path: Path) -> None:
    """The augment flags ride through TrainConfig into the training loop."""
    fit(
        GazeNet(pretrained=False),
        DataLoader(FakeGaze((0,)), batch_size=2),
        DataLoader(FakeGaze((1,)), batch_size=2),
        torch.device("cpu"),
        TrainConfig(
            epochs=1,
            batch_size=2,
            num_workers=0,
            augment_flip=True,
            augment_crop_jitter=True,
            augment_photometric=True,
        ),
        tmp_path,
        ConsoleLogger(tmp_path / "run.jsonl"),
    )
    assert len(read_jsonl(tmp_path / "run.jsonl")) > 0


def test_checkpoint_refuses_a_mismatched_dropout(tmp_path: Path) -> None:
    path = tmp_path / "last.pt"
    trained = GazeNet(pretrained=False, dropout=0.5)
    save_checkpoint(
        path, trained, torch.optim.AdamW(trained.parameters()), 0, 1.0, TrainConfig()
    )
    with pytest.raises(ValueError, match="dropout"):
        load_checkpoint(path, GazeNet(pretrained=False, dropout=0.1))


def test_checkpoint_refuses_a_mismatched_output_mode(tmp_path: Path) -> None:
    path = tmp_path / "last.pt"
    model = GazeNet(pretrained=False)
    save_checkpoint(
        path, model, torch.optim.AdamW(model.parameters()), 0, 1.0, TrainConfig()
    )
    with pytest.raises(ValueError, match="checkpoint is screen_fraction"):
        load_checkpoint(path, GazeNet(output_mode="gaze_direction", pretrained=False))


def test_fit_runs_checkpoints_and_logs(tmp_path: Path) -> None:
    train_loader = DataLoader(FakeGaze((0, 1)), batch_size=2)
    val_loader = DataLoader(FakeGaze((2,)), batch_size=2)

    best = fit(
        GazeNet(pretrained=False),
        train_loader,
        val_loader,
        torch.device("cpu"),
        TrainConfig(epochs=2, batch_size=2, num_workers=0),
        tmp_path,
        ConsoleLogger(tmp_path / "run.jsonl"),
    )

    assert best.count == 4
    assert (tmp_path / "last.pt").is_file()
    assert (tmp_path / "best.pt").is_file()
    assert (tmp_path / "best_metrics.json").is_file()
    # The log file is distinct from best_metrics.json, which fit overwrites.
    logged = read_jsonl(tmp_path / "run.jsonl")
    assert len(logged) == 3
    assert "val_p02" in logged[0]


def test_fit_exports_best_onnx_alongside_best_pt(tmp_path: Path) -> None:
    """Every best.pt rewrite also writes an onnx twin; last.pt gets none."""
    fit(
        GazeNet(pretrained=False),
        DataLoader(FakeGaze((0,)), batch_size=2),
        DataLoader(FakeGaze((1,)), batch_size=2),
        torch.device("cpu"),
        TrainConfig(epochs=1, batch_size=2, num_workers=0),
        tmp_path,
        ConsoleLogger(tmp_path / "run.jsonl"),
    )
    assert (tmp_path / "best.onnx").is_file()
    assert not (tmp_path / "last.onnx").exists()


def test_exported_onnx_matches_torch_across_batches(tmp_path: Path) -> None:
    """The onnx twin is dynamic-batch and numerically equal to the torch model.

    Exported from a model left in train mode: the export must work on a copy
    and leave the caller's weights and mode alone.
    """
    torch.manual_seed(0)
    model = GazeNet(pretrained=False)
    model.train()
    path = export_onnx(tmp_path / "best.onnx", model)
    assert model.training
    model.eval()

    proto = onnx.load(str(path))
    assert [value.name for value in proto.graph.input] == [
        "face",
        "left_eye",
        "right_eye",
    ]
    assert [value.name for value in proto.graph.output] == ["prediction"]
    session = onnxruntime.InferenceSession(
        str(path), providers=["CPUExecutionProvider"]
    )

    rng = np.random.default_rng(0)
    for batch in (1, 3):
        inputs = {
            "face": rng.standard_normal((batch, 3, 224, 224)).astype(np.float32),
            "left_eye": rng.standard_normal((batch, 3, 36, 60)).astype(np.float32),
            "right_eye": rng.standard_normal((batch, 3, 36, 60)).astype(np.float32),
        }
        with torch.no_grad():
            expected = model(
                torch.from_numpy(inputs["face"]),
                torch.from_numpy(inputs["left_eye"]),
                torch.from_numpy(inputs["right_eye"]),
            ).numpy()
        (got,) = session.run(None, inputs)
        # onnxruntime and torch accumulate fp32 convolutions in different
        # orders; on fixed inputs this stays around 2e-4 (the exported graph
        # itself is exact -- an fp64 export matches torch fp64 to 5e-15).
        # A real mistake -- wrong weights or wiring -- shows up at O(0.1).
        assert np.abs(got - expected).max() < 1e-3


def test_exported_onnx_without_eyes_takes_only_the_face(tmp_path: Path) -> None:
    model = GazeNet(pretrained=False, use_eyes=False).eval()
    path = export_onnx(tmp_path / "best.onnx", model)
    inputs = onnx.load(str(path)).graph.input
    assert [value.name for value in inputs] == ["face"]


def test_fit_resumes_from_the_last_checkpoint(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    train_loader = DataLoader(FakeGaze((0,)), batch_size=2)
    val_loader = DataLoader(FakeGaze((1,)), batch_size=2)

    args = (torch.device("cpu"),)
    fit(
        GazeNet(pretrained=False),
        train_loader,
        val_loader,
        *args,
        TrainConfig(epochs=1, batch_size=2, num_workers=0),
        tmp_path,
        ConsoleLogger(tmp_path / "run.jsonl"),
    )
    fit(
        GazeNet(pretrained=False),
        train_loader,
        val_loader,
        *args,
        TrainConfig(epochs=2, batch_size=2, num_workers=0),
        tmp_path,
        ConsoleLogger(tmp_path / "run2.jsonl"),
        resume=True,
    )
    assert "resumed from" in capsys.readouterr().out


def test_evaluate_reports_per_participant_error_and_loss() -> None:
    stats, loss = evaluate(
        GazeNet(pretrained=False),
        DataLoader(FakeGaze((3, 4)), batch_size=4),
        torch.device("cpu"),
    )
    assert stats.count == 8
    assert sorted(stats.per_participant) == ["p03", "p04"]
    assert loss > 0.0


def test_evaluate_works_in_gaze_direction_mode() -> None:
    """The mode needs 3-D direction targets from the dataset; this broke once."""
    stats, loss = evaluate(
        GazeNet(pretrained=False, output_mode="gaze_direction"),
        DataLoader(FakeGaze((3,)), batch_size=2),
        torch.device("cpu"),
    )
    assert stats.count == 4
    assert loss > 0.0


def test_fit_logs_val_loss_alongside_train_loss() -> None:
    """Both are mean-per-batch of the same criterion, so they are comparable."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        fit(
            GazeNet(pretrained=False),
            DataLoader(FakeGaze((0,)), batch_size=2),
            DataLoader(FakeGaze((1,)), batch_size=2),
            torch.device("cpu"),
            TrainConfig(epochs=1, batch_size=2, num_workers=0),
            Path(tmp),
            ConsoleLogger(Path(tmp) / "run.jsonl"),
        )
        records = read_jsonl(Path(tmp) / "run.jsonl")
    assert {"train_loss", "val_loss"} <= set(records[1])


def test_lr_schedule_none_keeps_the_lr_constant(tmp_path: Path) -> None:
    fit(
        GazeNet(pretrained=False),
        DataLoader(FakeGaze((0,)), batch_size=2),
        DataLoader(FakeGaze((1,)), batch_size=2),
        torch.device("cpu"),
        TrainConfig(epochs=2, batch_size=2, num_workers=0),
        tmp_path,
        ConsoleLogger(tmp_path / "run.jsonl"),
    )
    lrs = [record["lr"] for record in read_jsonl(tmp_path / "run.jsonl")]
    assert lrs == [1e-4, 1e-4, 1e-4]


def test_cosine_schedule_decays_the_lr(tmp_path: Path) -> None:
    fit(
        GazeNet(pretrained=False),
        DataLoader(FakeGaze((0,)), batch_size=2),
        DataLoader(FakeGaze((1,)), batch_size=2),
        torch.device("cpu"),
        TrainConfig(epochs=3, batch_size=2, num_workers=0, lr_schedule="cosine"),
        tmp_path,
        ConsoleLogger(tmp_path / "run.jsonl"),
    )
    # Baseline at the full lr; epochs 1..3 train at 1.0x, 0.75x, 0.25x.
    lrs = [record["lr"] for record in read_jsonl(tmp_path / "run.jsonl")]
    assert lrs == pytest.approx([1e-4, 1e-4, 0.75e-4, 0.25e-4])


def test_resume_continues_the_cosine_schedule(tmp_path: Path) -> None:
    """A resumed epoch trains with the same lr it would have had uninterrupted."""
    fit(
        GazeNet(pretrained=False),
        DataLoader(FakeGaze((0,)), batch_size=2),
        DataLoader(FakeGaze((1,)), batch_size=2),
        torch.device("cpu"),
        TrainConfig(epochs=1, batch_size=2, num_workers=0, lr_schedule="cosine"),
        tmp_path,
        ConsoleLogger(tmp_path / "run.jsonl"),
    )
    fit(
        GazeNet(pretrained=False),
        DataLoader(FakeGaze((0,)), batch_size=2),
        DataLoader(FakeGaze((1,)), batch_size=2),
        torch.device("cpu"),
        TrainConfig(epochs=3, batch_size=2, num_workers=0, lr_schedule="cosine"),
        tmp_path,
        ConsoleLogger(tmp_path / "run.jsonl"),
        resume=True,
    )
    lrs = {
        record["epoch"]: record["lr"] for record in read_jsonl(tmp_path / "run.jsonl")
    }
    assert lrs[2] == pytest.approx(0.75e-4)
    assert lrs[3] == pytest.approx(0.25e-4)


def test_early_stopping_halts_when_val_stops_improving(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Patience counts consecutive epochs with no improvement, then breaks."""
    # Seeded because `fit` does not: whether val plateaus within 20 epochs
    # depends on the init, and the unseeded stream is order-dependent.
    torch.manual_seed(0)
    fit(
        GazeNet(pretrained=False),
        DataLoader(FakeGaze((0,)), batch_size=2),
        DataLoader(FakeGaze((1,)), batch_size=2),
        torch.device("cpu"),
        TrainConfig(epochs=20, batch_size=2, num_workers=0, patience=1),
        tmp_path,
        ConsoleLogger(tmp_path / "run.jsonl"),
    )
    # Far fewer than 20 epochs ran, and the reason was printed.
    assert len(read_jsonl(tmp_path / "run.jsonl")) < 20
    assert "stopping early" in capsys.readouterr().out


def test_patience_none_runs_every_epoch(tmp_path: Path) -> None:
    fit(
        GazeNet(pretrained=False),
        DataLoader(FakeGaze((0,)), batch_size=2),
        DataLoader(FakeGaze((1,)), batch_size=2),
        torch.device("cpu"),
        TrainConfig(epochs=3, batch_size=2, num_workers=0, patience=None),
        tmp_path,
        ConsoleLogger(tmp_path / "run.jsonl"),
    )
    assert len(read_jsonl(tmp_path / "run.jsonl")) == 4


def test_checkpoint_refuses_a_mismatched_normalization(tmp_path: Path) -> None:
    """pretrained also selects input normalization, so a mismatch must fail loudly."""
    path = tmp_path / "last.pt"
    trained = GazeNet(pretrained=True)
    save_checkpoint(
        path, trained, torch.optim.AdamW(trained.parameters()), 0, 1.0, TrainConfig()
    )
    with pytest.raises(ValueError, match="input normalization"):
        load_checkpoint(path, GazeNet(pretrained=False))


def test_checkpoint_refuses_a_mismatched_eye_configuration(tmp_path: Path) -> None:
    path = tmp_path / "last.pt"
    trained = GazeNet(pretrained=False, use_eyes=True)
    save_checkpoint(
        path, trained, torch.optim.AdamW(trained.parameters()), 0, 1.0, TrainConfig()
    )
    with pytest.raises(ValueError, match="use_eyes"):
        load_checkpoint(path, GazeNet(pretrained=False, use_eyes=False))
