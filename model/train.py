"""Train the gaze model.

    uv run train.py

Splits are participant-disjoint: a subject never appears in both train and
eval, because generalizing to an unseen face is what we are measuring. MPII
splits by participant (--val/--test); GazeCapture has its own official
subject-disjoint train/val/test splits, which are used as-is.
"""

import argparse
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.gazecapture import GazeCapture
from src.data.mpiifacegaze import MPII
from src.training.calibration import fit_user_calibration
from src.training.logging import ConsoleLogger, WandbLogger, resolve_run_dir
from src.training.loop import (
    TrainConfig,
    collect_predictions,
    evaluate,
    fit,
    load_checkpoint,
)
from src.training.model import GazeNet

NUM_PARTICIPANTS = 15


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    data = parser.add_argument_group("data")
    data.add_argument(
        "--dataset",
        choices=["mpiifacegaze", "gazecapture"],
        default="mpiifacegaze",
        help="dataset to train on",
    )
    data.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="raw dataset root; defaults to data/<dataset>",
    )
    data.add_argument(
        "--val", type=int, default=13, help="MPII validation participant"
    )
    data.add_argument(
        "--test", type=int, default=14, help="MPII test participant"
    )

    model = parser.add_argument_group("model")
    model.add_argument(
        "--output-mode",
        choices=["screen_fraction", "gaze_direction"],
        default="screen_fraction",
    )
    model.add_argument("--no-eyes", action="store_true", help="ablate the eye towers")
    model.add_argument("--no-pretrained", action="store_true", help="random init")
    model.add_argument(
        "--dropout", type=float, default=0.1, help="head dropout; 0 disables it"
    )
    model.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="initialize from a previous run's checkpoint (weights only)",
    )

    augment = parser.add_argument_group("augmentation")
    augment.add_argument(
        "--flip",
        action="store_true",
        help="random horizontal flips (eye crops swap; labels transform to match)",
    )
    augment.add_argument(
        "--crop-jitter",
        action="store_true",
        help="random rotation/scale/translation of the crops",
    )
    augment.add_argument(
        "--photometric-jitter",
        action="store_true",
        help="random brightness/contrast",
    )

    optim = parser.add_argument_group("optimization")
    optim.add_argument("--epochs", type=int, default=10)
    optim.add_argument("--lr", type=float, default=1e-4)
    optim.add_argument(
        "--lr-schedule",
        choices=["cosine", "none"],
        default="cosine",
        help="decay the lr over --epochs (cosine) or hold it constant",
    )
    optim.add_argument("--weight-decay", type=float, default=0.01)
    optim.add_argument("--batch-size", type=int, default=256)
    optim.add_argument("--workers", type=int, default=8)
    optim.add_argument("--seed", type=int, default=0)
    optim.add_argument(
        "--patience",
        type=int,
        default=None,
        help="stop after this many epochs with no improvement",
    )

    calibration = parser.add_argument_group("calibration")
    calibration.add_argument(
        "--ridge-lambda",
        type=float,
        default=0.1,
        help="ridge penalty for the per-user calibration."
    )

    run = parser.add_argument_group("run")
    run.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="auto | cpu | cuda | mps",
    )
    run.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints"),
        help="parent directory; each run gets a <timestamp> subdirectory",
    )
    run.add_argument("--resume", action="store_true")
    run.add_argument(
        "--wandb",
        metavar="PROJECT",
        help="wandb project this run is tracked in",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.manual_seed(args.seed)

    data_dirs = {"mpiifacegaze": "MPIIFaceGaze", "gazecapture": "GazeCapture"}
    args.data_root = args.data_root or (
        Path(__file__).resolve().parents[1] / "data" / data_dirs[args.dataset]
    )

    splits: dict[str, Any]
    if args.dataset == "mpiifacegaze":
        for name, participant in (("--val", args.val), ("--test", args.test)):
            if not 0 <= participant < NUM_PARTICIPANTS:
                raise SystemExit(f"{name} must be in 0..{NUM_PARTICIPANTS - 1}")
        if args.val == args.test:
            raise SystemExit("--val and --test must be different participants")

        train_ids = [
            p for p in range(NUM_PARTICIPANTS) if p not in (args.val, args.test)
        ]
        splits = {"train": train_ids, "val": args.val, "test": args.test}
    else:
        splits = {"train": "train", "val": "val", "test": "test"}

    if args.weights is not None and args.resume:
        raise SystemExit("--weights and --resume are mutually exclusive")

    run_dir = resolve_run_dir(args.checkpoint_dir, args.resume)
    run_dir.mkdir(parents=True, exist_ok=True)

    def loader(role: str, *, shuffle: bool) -> DataLoader[Any]:
        if args.dataset == "mpiifacegaze":
            dataset: Any = MPII(splits[role], args.data_root)
        else:
            dataset = GazeCapture(splits[role], args.data_root)
        return DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.workers,
            drop_last=shuffle,
            persistent_workers=args.workers > 0,
        )

    if args.device != "auto":
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = GazeNet(
        output_mode=args.output_mode,
        pretrained=not args.no_pretrained,
        use_eyes=not args.no_eyes,
        dropout=args.dropout,
    )
    print(f"{model.parameter_count() / 1e6:.2f}M parameters on {device}")

    if args.weights is not None:
        load_checkpoint(args.weights, model)
        print(f"initialized weights from {args.weights}")

    config = TrainConfig(
        epochs=args.epochs,
        learning_rate=args.lr,
        lr_schedule=args.lr_schedule,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        num_workers=args.workers,
        seed=args.seed,
        patience=args.patience,
        augment_flip=args.flip,
        augment_crop_jitter=args.crop_jitter,
        augment_photometric=args.photometric_jitter,
    )

    logger = (
        WandbLogger(project=args.wandb, config=vars(args))
        if args.wandb
        else ConsoleLogger(run_dir / "run.jsonl")
    )
    best = fit(
        model,
        loader("train", shuffle=True),
        loader("val", shuffle=False),
        device,
        config,
        run_dir,
        logger,
        resume=args.resume,
    )
    print(f"\nbest val mean {best.mean:.5f} (median {best.median:.5f})")
    worst = best.worst_participant()
    if worst is not None:
        print(f"worst participant: {worst[0]} at {worst[1].mean:.5f}")

    # Recover the best model before running test. The test set is scored
    # twice: raw, and with a per-user ridge correction that main fits here on
    # each test subject's extreme samples before handing it to evaluate.
    load_checkpoint(run_dir / "best.pt", model)
    test_loader = loader("test", shuffle=False)
    raw_stats, test_loss = evaluate(model, test_loader, device)
    print(
        f"test mean {raw_stats.mean:.5f} (loss {test_loss:.5f}) over {raw_stats.count} samples"
    )

    predictions, targets, screen_labels, subjects = collect_predictions(
        model, test_loader, device
    )
    fits = fit_user_calibration(
        predictions.numpy(),
        targets.numpy(),
        subjects,
        selection_labels=screen_labels.numpy(),
        lamb=args.ridge_lambda,
    )
    calibrated_stats, _ = evaluate(model, test_loader, device, calibration=fits)
    improvement = (
        f" ({100 * (1 - calibrated_stats.mean / raw_stats.mean):+.1f}%)"
        if raw_stats.mean > 0
        else ""
    )
    print(
        f"test mean with per-user ridge calibration:"
        f" {calibrated_stats.mean:.5f}{improvement}"
    )

    logger.summarize({
        "best_val_mean": best.mean,
        "best_val_median": best.median,
        **(
            {
                "worst_participant": worst[0],
                "worst_participant_mean": worst[1].mean,
            }
            if worst is not None
            else {}
        ),
        "test_mean": raw_stats.mean,
        "test_median": raw_stats.median,
        "test_p95": raw_stats.p95,
        "test_loss": test_loss,
        "test_count": raw_stats.count,
        "test_mean_calibrated": calibrated_stats.mean,
        "test_median_calibrated": calibrated_stats.median,
        "test_p95_calibrated": calibrated_stats.p95,
    })
    logger.finish()


if __name__ == "__main__":
    main()
