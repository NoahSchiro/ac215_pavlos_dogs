"""Train and evaluate loops."""

import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.utils import prepare_images
from src.training.augment import augment_batch
from src.training.config import TrainConfig
from src.training.export import export_onnx, load_checkpoint, save_checkpoint
from src.training.logging import RunLogger
from src.training.metrics import (
    ErrorStats,
    angular_error_deg,
    screen_fraction_error,
    summarize,
)
from src.training.model import GazeNet


def forward_batch(
    model: GazeNet,
    batch: dict[str, Any],
    device: torch.device,
    config: TrainConfig | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Move one collated batch to device and run it. Returns (prediction, target).

    `batch` is a dict of stacked tensors, not a list of per-sample dicts:
    torch's default collate turns a list of dicts into a dict of batches.
    `config` enables the training-time augmentations; evaluation passes
    nothing and always sees the raw crops.
    """
    if config is not None:
        batch = augment_batch(
            batch,
            device,
            flip=config.augment_flip,
            crop_jitter=config.augment_crop_jitter,
            photometric=config.augment_photometric,
        )
    face = prepare_images(batch["face"], device, imagenet=model.pretrained)
    if model.use_eyes:
        prediction = model(
            face,
            prepare_images(batch["left_eye"], device, imagenet=model.pretrained),
            prepare_images(batch["right_eye"], device, imagenet=model.pretrained),
        )
    else:
        prediction = model(face)
    key = "direction" if model.output_mode == "gaze_direction" else "label"
    if key not in batch:
        raise ValueError(
            f"dataset samples lack {key!r}; a cache built before direction"
            " labels existed must be rebuilt"
        )
    return prediction, batch[key].to(device, non_blocking=True).float()


def compute_loss(
    model: GazeNet,
    criterion: nn.Module,
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Apply the criterion, handling CosineEmbeddingLoss's extra target argument."""
    if model.output_mode == "gaze_direction":
        ones = torch.ones(prediction.shape[0], device=prediction.device)
        return criterion(prediction, target, ones)
    return criterion(prediction, target)


def build_criterion(model: GazeNet) -> nn.Module:
    """Cosine distance for directions, squared error for screen coordinates."""
    if model.output_mode == "gaze_direction":
        return nn.CosineEmbeddingLoss()
    return nn.MSELoss()


def train_one_epoch(
    model: GazeNet,
    loader: DataLoader[Any],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    config: TrainConfig,
) -> float:
    """One pass over the training set. Returns mean loss per batch."""
    model.train()
    total = 0.0
    for batch in tqdm(loader, desc="train", leave=False):
        optimizer.zero_grad(set_to_none=True)
        prediction, target = forward_batch(model, batch, device, config)
        loss = compute_loss(model, criterion, prediction, target)
        loss.backward()
        if config.grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        total += float(loss.item())
    return total / max(len(loader), 1)


@torch.no_grad()
def evaluate(
    model: GazeNet,
    loader: DataLoader[Any],
    device: torch.device,
    criterion: nn.Module | None = None,
) -> tuple[ErrorStats, float]:
    """Error over the whole loader, broken down per participant.

    Returns (stats, mean loss per batch). The loss is the same quantity
    train_one_epoch reports, so the two are directly comparable: validation
    loss rising while training loss falls is the overfitting signal.
    """
    model.eval()
    criterion = criterion if criterion is not None else build_criterion(model)
    errors: list[torch.Tensor] = []
    subjects: list[str] = []
    total = 0.0
    for batch in tqdm(loader, desc="eval", leave=False):
        prediction, target = forward_batch(model, batch, device)
        total += float(compute_loss(model, criterion, prediction, target).item())
        if model.output_mode == "gaze_direction":
            errors.append(angular_error_deg(prediction, target).cpu())
        else:
            errors.append(screen_fraction_error(prediction, target).cpu())
        # sample_id looks like "<dataset>/<subject>/..."
        subjects.extend(sample_id.split("/")[1] for sample_id in batch["sample_id"])
    if not errors:
        raise ValueError("evaluation loader produced no batches")
    return summarize(torch.cat(errors), subjects), total / len(loader)


def fit(
    model: GazeNet,
    train_loader: DataLoader[Any],
    val_loader: DataLoader[Any],
    device: torch.device,
    config: TrainConfig,
    checkpoint_dir: Path,
    logger: RunLogger,
    resume: bool = False,
) -> ErrorStats:
    """Train for `config.epochs`, returning the best validation stats."""
    model.to(device)

    decay = [p for p in model.parameters() if p.ndim > 1]
    no_decay = [p for p in model.parameters() if p.ndim <= 1]
    optimizer = torch.optim.AdamW([
        {"params": decay, "weight_decay": config.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=config.learning_rate)
    criterion = build_criterion(model)

    start_epoch = 0
    best_error = float("inf")
    last_path = checkpoint_dir / "last.pt"
    if resume and last_path.is_file():
        start_epoch, best_error = load_checkpoint(last_path, model, optimizer)
        print(f"resumed from {last_path} at epoch {start_epoch}")

    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
    if config.lr_schedule == "cosine":
        t_max = max(config.epochs, 1)
        scheduler = CosineAnnealingLR(optimizer, T_max=t_max)
        completed = max(start_epoch - 1, 0)
        scheduler.base_lrs = [config.learning_rate] * len(optimizer.param_groups)
        for group in optimizer.param_groups:
            group["lr"] = config.learning_rate
        scheduler.last_epoch = completed
        for group, base in zip(optimizer.param_groups, scheduler.base_lrs):
            group["lr"] = (
                scheduler.eta_min
                + (base - scheduler.eta_min)
                * (1 + math.cos(math.pi * completed / t_max))
                / 2
            )
    elif config.lr_schedule != "none":
        raise ValueError(
            f"lr_schedule must be 'none' or 'cosine', got {config.lr_schedule!r}"
        )

    # Take a baseline before any training
    stats, val_loss = evaluate(model, val_loader, device, criterion)
    last_stats: ErrorStats = stats
    best_stats: ErrorStats | None = None
    stale = 0
    logger.log({
        "epoch": 0,
        "lr": optimizer.param_groups[0]["lr"],
        "val_loss": val_loss,
        "val_mean": stats.mean,
        "val_median": stats.median,
        "val_p95": stats.p95,
        **{
            f"val_{subject}": s.mean
            for subject, s in stats.per_participant.items()
        },
    })

    for epoch in range(max(start_epoch, 1), config.epochs+1):
        started = time.monotonic()
        lr = optimizer.param_groups[0]["lr"]
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, config
        )
        if scheduler is not None:
            scheduler.step()
        stats, val_loss = evaluate(model, val_loader, device, criterion)
        last_stats = stats

        improved = stats.mean < best_error
        stale = 0 if improved else stale + 1
        if improved:
            best_error = stats.mean
            best_stats = stats
            save_checkpoint(
                checkpoint_dir / "best.pt", model, optimizer, epoch, best_error, config
            )
            export_onnx(checkpoint_dir / "best.onnx", model)
        save_checkpoint(last_path, model, optimizer, epoch, best_error, config)

        logger.log({
            "epoch": epoch,
            "lr": lr,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_mean": stats.mean,
            "val_median": stats.median,
            "val_p95": stats.p95,
            "seconds": time.monotonic() - started,
            **{
                f"val_{subject}": s.mean
                for subject, s in stats.per_participant.items()
            },
        })

        if config.patience is not None and stale >= config.patience:
            print(
                f"stopping early: no improvement for {stale} epochs"
                f" (best {best_error:.5f})"
            )
            break

    if best_stats is None:
        # A resumed run that never beat the earlier best is a normal outcome;
        # report the final epoch and leave the previous best_metrics.json.
        return last_stats

    (checkpoint_dir / "best_metrics.json").write_text(
        json.dumps(asdict(best_stats), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return best_stats
