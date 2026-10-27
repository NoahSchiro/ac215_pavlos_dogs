"""Hyperparameters for one training run, recorded with its checkpoint."""

from dataclasses import dataclass


@dataclass
class TrainConfig:
    """Hyperparameters for one run, separate from how the run was invoked.

    This is what `save_checkpoint` records, so it holds the things that
    determine the resulting weights and none of the things that do not
    (device, paths, --resume). It also lets `fit` be called without
    argparse, which the tests rely on.

    `batch_size` and `num_workers` are not read by `fit` -- the loaders are
    built by the caller -- but are recorded so a checkpoint says what
    produced it.
    """

    epochs: int = 30
    learning_rate: float = 1e-4
    lr_schedule: str = "none"
    weight_decay: float = 1e-4
    batch_size: int = 256
    num_workers: int = 8
    seed: int = 0
    grad_clip: float | None = 1.0
    patience: int | None = None
    """Stop after this many epochs with no improvement. None disables it."""
    augment_flip: bool = False
    augment_crop_jitter: bool = False
    augment_photometric: bool = False
    # Per-batch augmentation switches, applied on device during training
    # only (see src/training/augment.py); recorded with the checkpoint.
