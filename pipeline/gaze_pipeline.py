"""Vertex AI pipeline: preprocess, train, evaluate, publish.

Each step runs the container the repository already builds
(`model/Dockerfile`) with a different command. No training logic is
duplicated here -- the pipeline decides only what runs, in what order, on
what machine. It does name train.py's flags, so renaming one there needs
the same change here.

Vertex mounts Cloud Storage under /gcs, so `gs://bucket/path` is readable
inside a step as `/gcs/bucket/path`. That is how the dataset reaches
preprocessing and how checkpoints outlive the run.

Steps are ordered with `.after()` rather than by passing artifacts between
them: the cache and the checkpoints live in the bucket, so the only thing
the DAG needs to express is sequence.

Each pipeline run gets its own folder, `<checkpoint_dir>/<run_id>/`, and
train, evaluate and publish only ever look inside it. Without that, evaluate
and publish would pick "the newest run in the bucket", which is another run's
model whenever two pipelines (or a pipeline and a manual upload) overlap.
"""

from kfp import dsl
from kfp.dsl.graph_component import GraphComponent

# What --compile-only builds against when no image is given. submit.py
# refuses to submit it, so it can never reach Vertex.
PLACEHOLDER_IMAGE = "us-east1-docker.pkg.dev/PROJECT/gaze/gaze-model:latest"

# A run id names a folder, so it must be one plain path segment.
_CHECK_RUN_ID = """case "$run_id" in
  ''|*[!A-Za-z0-9._-]*|.|..) echo "invalid run id: '$run_id'" >&2; exit 1 ;;
esac
"""

_PREPROCESS_SH = """set -euo pipefail
if [ -e "$0/MPIIFaceGaze_wd" ]; then
  echo "cache already present under $0; skipping"
  exit 0
fi
echo "building webdataset cache under $0"
uv run python -c "from src.data.mpiifacegaze import MPII; MPII(list(range(15)), '$0')"
"""

_EVALUATE_SH = (
    """set -euo pipefail
run_id="$1"; out="$2"
"""
    + _CHECK_RUN_ID
    + """run_root="$0/$run_id"
# Only this pipeline run's folder. A retried train step can leave more than
# one timestamped attempt in it; the newest is the one that finished.
# `|| true`: with pipefail, an empty folder would otherwise exit here silently.
latest="$(ls -d "$run_root"/*/ 2>/dev/null | sort | tail -1 || true)"
if [ -z "$latest" ]; then echo "no run directory under $run_root" >&2; exit 1; fi
echo "reporting metrics from $latest"
mkdir -p "$(dirname "$out")"
cp "$latest/best_metrics.json" "$out"
cat "$out"
"""
)


_TRAIN_SH = (
    """set -euo pipefail
# $0 is the key file, so only $1..$3 shift off; the rest are train.py flags.
key_file="$0"; project="$1"; checkpoints="$2"; run_id="$3"; shift 3
"""
    + _CHECK_RUN_ID
    + """# train.py makes a <timestamp> folder under --checkpoint-dir, so this run's
# weights land in <checkpoints>/<run_id>/<timestamp>/.
run_root="$checkpoints/$run_id"

# DataLoader workers hand batches to the trainer through /dev/shm. Containers
# often get a small one (Docker's default is 64MB), and running out of it
# kills training mid-epoch with "No space left on device". Measure it, and
# below 1GB load data in the main process instead (slower, but needs none).
shm="${SHM_DIR:-/dev/shm}"
shm_kb="$(df -Pk "$shm" 2>/dev/null | tail -1 | tr -s ' ' | cut -d' ' -f2 || true)"
workers=""
case "$shm_kb" in
  ''|*[!0-9]*) workers="--workers 0"; echo "no usable $shm: --workers 0" ;;
  *) if [ "$shm_kb" -lt 1048576 ]; then
       workers="--workers 0"; echo "$shm is only ${shm_kb}KB: --workers 0"
     else
       echo "$shm is ${shm_kb}KB: keeping parallel data loading"
     fi ;;
esac

if [ -z "$project" ]; then
  exec uv run train.py --checkpoint-dir "$run_root" "$@" $workers
fi
if [ ! -s "$key_file" ]; then
  echo "W&B project '$project' requested but there is no API key at $key_file" >&2
  echo "(upload one with: gcloud storage cp <file> gs://${key_file#/gcs/})" >&2
  exit 1
fi
WANDB_API_KEY="$(cat "$key_file")"
export WANDB_API_KEY
exec uv run train.py --wandb "$project" --checkpoint-dir "$run_root" "$@" $workers
"""
)

_PUBLISH_SH = (
    """set -euo pipefail
run_id="$1"; bucket="$2"; out="$3"
"""
    + _CHECK_RUN_ID
    + """run_root="$0/$run_id"
root="${GCS_MOUNT:-/gcs}/$bucket"
# Only this pipeline run's folder, so a concurrent run can't be published by mistake.
# `|| true`: with pipefail, an empty folder would otherwise exit here silently.
latest="$(ls -d "$run_root"/*/ 2>/dev/null | sort | tail -1 || true)"
latest="${latest%/}"
if [ -z "$latest" ] || [ ! -f "$latest/best.onnx" ]; then
  echo "no best.onnx in this run's folder $run_root" >&2; exit 1
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
)


def gcs_mount(uri: str) -> str:
    """`gs://bucket/path` -> `/gcs/bucket/path`, the path Vertex mounts it at."""
    if not uri.startswith("gs://"):
        raise ValueError(f"expected a gs:// uri, got {uri!r}")
    return "/gcs/" + uri[len("gs://") :]


def make_pipeline(image: str) -> GraphComponent:
    """The pipeline, with every step running `image`.

    The image is an argument here, not a pipeline parameter: KFP writes it
    into each step's container spec when the steps are defined. Defining them
    inside this function makes that an ordinary argument, rather than state
    that has to be set before the module is imported.
    """

    @dsl.container_component
    def preprocess_op(data_root: str) -> dsl.ContainerSpec:
        """Build the webdataset cache unless it already sits beside the raw data.

        Idempotent on purpose: the cache lives in the bucket, so rerunning the
        pipeline skips a MediaPipe pass over every image instead of repeating it.
        """
        return dsl.ContainerSpec(
            image=image, command=["bash", "-c"], args=[_PREPROCESS_SH, data_root]
        )

    @dsl.container_component
    def train_op(
        data_root: str,
        checkpoint_dir: str,
        run_id: str,
        epochs: int,
        val_participant: int,
        test_participant: int,
        wandb_project: str,
        wandb_key_path: str,
    ) -> dsl.ContainerSpec:
        """Run train.py, with its flags spelled out here (rename one, rename both).

        W&B needs an API key, and a Vertex container has no `wandb login`. The key
        is read at runtime from a file in the private data bucket rather than
        passed as a parameter, because parameters are stored in the pipeline spec
        and shown in the Vertex console. A requested project with no key fails
        before training starts, not after. An empty wandb_project skips W&B.
        """
        return dsl.ContainerSpec(
            image=image,
            command=["bash", "-c"],
            args=[
                _TRAIN_SH,
                wandb_key_path,
                wandb_project,
                # The script passes --checkpoint-dir <checkpoint_dir>/<run_id>.
                checkpoint_dir,
                run_id,
                "--data-root",
                data_root,
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
    def evaluate_op(
        checkpoint_dir: str,
        run_id: str,
        metrics: dsl.OutputPath(str),  # type: ignore[valid-type]
    ) -> dsl.ContainerSpec:
        """Lift the run's best metrics out as a pipeline artifact.

        `fit` already writes best_metrics.json beside the checkpoints. Copying it
        to an output path is what makes the number visible in the Vertex run
        graph instead of only in the bucket.
        """
        return dsl.ContainerSpec(
            image=image,
            command=["bash", "-c"],
            args=[_EVALUATE_SH, checkpoint_dir, run_id, metrics],
        )

    @dsl.container_component
    def publish_op(
        checkpoint_dir: str,
        run_id: str,
        models_bucket: str,
        model_url: dsl.OutputPath(str),  # type: ignore[valid-type]
    ) -> dsl.ContainerSpec:
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
            image=image,
            command=["bash", "-c"],
            args=[_PUBLISH_SH, checkpoint_dir, run_id, models_bucket, model_url],
        )

    @dsl.pipeline(
        name="gaze-training",
        description="Preprocess MPIIFaceGaze, train the gaze model, report and publish it.",
    )
    def gaze_pipeline(
        data_root: str = "/gcs/ac215-gaze-data-rh/MPIIFaceGaze",
        checkpoint_dir: str = "/gcs/ac215-gaze-data-rh/checkpoints",
        # submit.py gives every submission a fresh id; see the module docstring.
        run_id: str = "manual",
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
            run_id=run_id,
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

        evaluate = evaluate_op(checkpoint_dir=checkpoint_dir, run_id=run_id)
        evaluate.after(train)
        evaluate.set_display_name("evaluate (publish metrics)")

        # Only a run that produced its metrics gets published.
        publish = publish_op(
            checkpoint_dir=checkpoint_dir, run_id=run_id, models_bucket=models_bucket
        )
        publish.after(evaluate)
        publish.set_display_name("publish model for the UI")

    return gaze_pipeline
