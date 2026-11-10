"""Vertex AI pipeline: preprocess, train, evaluate, publish.

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


_TRAIN_SH = """set -euo pipefail
key_file="$0"; project="$1"; shift  # $0 is the key file, so only $1 shifts off
if [ -z "$project" ]; then
  exec uv run train.py "$@"
fi
if [ ! -s "$key_file" ]; then
  echo "W&B project '$project' requested but there is no API key at $key_file" >&2
  echo "(upload one with: gcloud storage cp <file> gs://${key_file#/gcs/})" >&2
  exit 1
fi
WANDB_API_KEY="$(cat "$key_file")"
export WANDB_API_KEY
exec uv run train.py --wandb "$project" "$@"
"""

_PUBLISH_SH = """set -euo pipefail
checkpoints="$0"; bucket="$1"; out="$2"
root="${GCS_MOUNT:-/gcs}/$bucket"
latest="$(ls -d "$checkpoints"/*/ 2>/dev/null | sort | tail -1)"
latest="${latest%/}"
if [ -z "$latest" ] || [ ! -f "$latest/best.onnx" ]; then
  echo "no best.onnx in the newest run under $checkpoints" >&2; exit 1
fi
version="$(basename "$latest")"
dest="$root/models/$version"
if [ -e "$dest/gazenet.onnx" ] && ! cmp -s "$latest/best.onnx" "$dest/gazenet.onnx"; then
  echo "$dest holds a different model; published versions are immutable" >&2; exit 1
fi
mkdir -p "$dest"
cp "$latest/best.onnx" "$dest/gazenet.onnx"
if [ -f "$latest/best_metrics.json" ]; then cp "$latest/best_metrics.json" "$dest/metrics.json"; fi
url="https://storage.googleapis.com/$bucket/models/$version/gazenet.onnx"
mkdir -p "$(dirname "$out")"
printf '%s' "$url" > "$out"
echo "published MODEL_URL=$url"
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
    wandb_key_path: str,
):
    """Run train.py. Flags map one-to-one, so the CLI stays the source of truth.

    W&B needs an API key, and a Vertex container has no `wandb login`. The key
    is read at runtime from a file in the private data bucket rather than
    passed as a parameter, because parameters are stored in the pipeline spec
    and shown in the Vertex console. A requested project with no key fails
    before training starts, not after. An empty wandb_project skips W&B.
    """
    return dsl.ContainerSpec(
        image=IMAGE,
        command=["bash", "-c"],
        args=[
            _TRAIN_SH,
            wandb_key_path,
            wandb_project,
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


@dsl.container_component
def publish_op(
    checkpoint_dir: str,
    models_bucket: str,
    model_url: dsl.OutputPath(str),  # type: ignore[valid-type]
):
    """Publish the run's best.onnx where the UI can load it.

    Copies it to gs://<models_bucket>/models/<run>/gazenet.onnx with the run's
    metrics beside it -- the layout deploy/publish-model.sh writes too -- and
    outputs the public URL, which is the MODEL_URL to deploy the UI with
    (deploy/deploy-ui.sh). That bucket is public-read for browsers; the
    checkpoints and data stay private.

    A version is never overwritten with a different model. Republishing the
    same bytes succeeds, so a retried step does not fail on its own output.
    """
    return dsl.ContainerSpec(
        image=IMAGE,
        command=["bash", "-c"],
        args=[_PUBLISH_SH, checkpoint_dir, models_bucket, model_url],
    )


@dsl.pipeline(
    name="gaze-training",
    description="Preprocess MPIIFaceGaze, train the gaze model, report and publish it.",
)
def gaze_pipeline(
    data_root: str = "/gcs/ac215-gaze-data-rh/MPIIFaceGaze",
    checkpoint_dir: str = "/gcs/ac215-gaze-data-rh/checkpoints",
    epochs: int = 10,
    val_participant: int = 13,
    test_participant: int = 14,
    wandb_project: str = "gaze",
    wandb_key_path: str = "/gcs/ac215-gaze-data-rh/secrets/wandb_api_key",
    models_bucket: str = "ac215-gaze-models",
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
        wandb_key_path=wandb_key_path,
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

    # Only a run that produced its metrics gets published.
    publish = publish_op(checkpoint_dir=checkpoint_dir, models_bucket=models_bucket)
    publish.after(evaluate)
    publish.set_display_name("publish model for the UI")
