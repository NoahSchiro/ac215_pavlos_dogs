"""Vertex AI pipeline: preprocess, train, evaluate.

Each step runs the container the repository already builds
(`model/Dockerfile`) with a different command. No training logic is
duplicated here -- the pipeline decides only what runs, in what order, on
what machine, so a change to `train.py` needs no change in this file.

Vertex mounts Cloud Storage under /gcs, so `gs://bucket/path` is readable
inside a step as `/gcs/bucket/path`. That is how the dataset reaches
preprocessing and how checkpoints outlive the run.

Steps are ordered with `.after()` rather than by passing artifacts between
them: the cache and the checkpoints live in the bucket, so the only thing
the DAG needs to express is sequence.
"""

import os

from kfp import dsl

# KFP resolves the image at compile time -- it belongs to the deployment
# config rather than the DAG -- so it cannot be a pipeline parameter.
# submit.py sets GAZE_IMAGE before importing this module to retarget a run.
IMAGE = os.environ.get(
    "GAZE_IMAGE", "us-east1-docker.pkg.dev/PROJECT/gaze/gaze-model:latest"
)

_PREPROCESS_SH = """set -euo pipefail
if [ -e "$0/MPIIFaceGaze_wd" ]; then
  echo "cache already present under $0; skipping"
  exit 0
fi
echo "building webdataset cache under $0"
uv run python -c "from src.data.mpiifacegaze import MPII; MPII(list(range(15)), '$0')"
"""

_EVALUATE_SH = """set -euo pipefail
latest="$(ls -d "$0"/*/ 2>/dev/null | sort | tail -1)"
if [ -z "$latest" ]; then echo "no run directory under $0" >&2; exit 1; fi
echo "reporting metrics from $latest"
mkdir -p "$(dirname "$1")"
cp "$latest/best_metrics.json" "$1"
cat "$1"
"""


def gcs_mount(uri: str) -> str:
    """`gs://bucket/path` -> `/gcs/bucket/path`, the path Vertex mounts it at."""
    if not uri.startswith("gs://"):
        raise ValueError(f"expected a gs:// uri, got {uri!r}")
    return "/gcs/" + uri[len("gs://") :]


@dsl.container_component
def preprocess_op(data_root: str):
    """Build the webdataset cache unless it already sits beside the raw data.

    Idempotent on purpose: the cache lives in the bucket, so rerunning the
    pipeline skips a MediaPipe pass over every image instead of repeating it.
    """
    return dsl.ContainerSpec(
        image=IMAGE, command=["bash", "-c"], args=[_PREPROCESS_SH, data_root]
    )


@dsl.container_component
def train_op(
    data_root: str,
    checkpoint_dir: str,
    epochs: int,
    val_participant: int,
    test_participant: int,
    wandb_project: str,
):
    """Run train.py. Flags map one-to-one, so the CLI stays the source of truth."""
    return dsl.ContainerSpec(
        image=IMAGE,
        command=["uv", "run", "train.py"],
        args=[
            "--data-root",
            data_root,
            "--checkpoint-dir",
            checkpoint_dir,
            "--epochs",
            epochs,
            "--val",
            val_participant,
            "--test",
            test_participant,
            "--wandb",
            wandb_project,
            # `auto` picks cuda on an accelerator machine type, cpu otherwise.
            "--device",
            "auto",
        ],
    )


@dsl.container_component
def evaluate_op(checkpoint_dir: str, metrics: dsl.OutputPath(str)):  # type: ignore[valid-type]
    """Lift the run's best metrics out as a pipeline artifact.

    `fit` already writes best_metrics.json beside the checkpoints. Copying it
    to an output path is what makes the number visible in the Vertex run
    graph instead of only in the bucket.
    """
    return dsl.ContainerSpec(
        image=IMAGE,
        command=["bash", "-c"],
        args=[_EVALUATE_SH, checkpoint_dir, metrics],
    )


@dsl.pipeline(
    name="gaze-training",
    description="Preprocess MPIIFaceGaze, train the gaze model, publish metrics.",
)
def gaze_pipeline(
    data_root: str = "/gcs/ac215-gaze-data-rh/MPIIFaceGaze",
    checkpoint_dir: str = "/gcs/ac215-gaze-data-rh/checkpoints",
    epochs: int = 10,
    val_participant: int = 13,
    test_participant: int = 14,
    wandb_project: str = "gaze-maze",
) -> None:
    preprocess = preprocess_op(data_root=data_root)
    preprocess.set_display_name("preprocess (build cache)")

    train = train_op(
        data_root=data_root,
        checkpoint_dir=checkpoint_dir,
        epochs=epochs,
        val_participant=val_participant,
        test_participant=test_participant,
        wandb_project=wandb_project,
    )
    train.after(preprocess)
    train.set_display_name("train")
    # DataLoader workers pass tensors through shared memory; a batch of
    # 224x224 crops needs considerably more than a small default.
    train.set_memory_limit("32G")
    train.set_cpu_limit("8")

    evaluate = evaluate_op(checkpoint_dir=checkpoint_dir)
    evaluate.after(train)
    evaluate.set_display_name("evaluate (publish metrics)")
