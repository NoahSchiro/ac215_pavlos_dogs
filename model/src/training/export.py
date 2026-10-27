"""Checkpoint save/load, and ONNX export of a trained GazeNet for inference
outside torch (e.g. the browser)."""

import copy
import warnings
from dataclasses import asdict
from pathlib import Path

import torch

from src.training.model import GazeNet
from src.training.loop import TrainConfig


def save_checkpoint(
    path: Path,
    model: GazeNet,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_error: float,
    config: TrainConfig,
) -> None:
    """Everything needed to resume, including on another machine."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "best_error": best_error,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "output_mode": model.output_mode,
            "use_eyes": model.use_eyes,
            "dropout": model.dropout,
            "pretrained": model.pretrained,
            "config": asdict(config),
        },
        path,
    )


def load_checkpoint(
    path: Path, model: GazeNet, optimizer: torch.optim.Optimizer | None = None
) -> tuple[int, float]:
    """Restore weights (and optimizer, if given). Returns (last_epoch, best)."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state["output_mode"] != model.output_mode:
        raise ValueError(
            f"checkpoint is {state['output_mode']}, model is {model.output_mode}"
        )
    # `pretrained` selects whether inputs get ImageNet normalization, so
    # loading weights into a model configured the other way produces much
    # worse predictions and no error. Fail loudly instead.
    if "pretrained" in state and state["pretrained"] != model.pretrained:
        raise ValueError(
            f"checkpoint was trained with pretrained={state['pretrained']},"
            f" model has pretrained={model.pretrained};"
            " the input normalization would not match"
        )
    if state.get("use_eyes") is not None and state["use_eyes"] != model.use_eyes:
        raise ValueError(
            f"checkpoint has use_eyes={state['use_eyes']},"
            f" model has use_eyes={model.use_eyes}"
        )
    if "dropout" in state and float(state["dropout"]) != model.dropout:
        raise ValueError(
            f"checkpoint was trained with dropout={state['dropout']},"
            f" model has dropout={model.dropout};"
            " the training dynamics would not match"
        )
    model.load_state_dict(state["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer_state"])
    return int(state["epoch"]), float(state["best_error"])


def export_onnx(path: Path, model: GazeNet) -> Path:
    """Write `model` as a self-contained ONNX file at `path` and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    exported = copy.deepcopy(model).cpu().eval()
    face_size = 244
    face = torch.zeros(1, 3, face_size, face_size)
    sample: tuple[torch.Tensor, ...]
    dynamic_shapes: tuple[dict[int, str], ...]
    names: tuple[str, ...]
    if model.use_eyes:
        eye_size = (36, 60)
        eyes = torch.zeros(1, 3, *eye_size)
        sample = (face, eyes, eyes)
        dynamic_shapes = ({0: "batch"}, {0: "batch"}, {0: "batch"})
        names = ("face", "left_eye", "right_eye")
    else:
        sample = (face,)
        dynamic_shapes = ({0: "batch"},)
        names = ("face",)

    with warnings.catch_warnings():
        # This process emits some irrelvant warnings
        warnings.filterwarnings(
            "ignore",
            message="# The axis name: batch will not be used",
            category=UserWarning,
        )
        torch.onnx.export(
            exported,
            sample,
            str(path),
            input_names=names,
            output_names=("prediction",),
            dynamic_shapes=dynamic_shapes,
            external_data=False,
        )
    return path
