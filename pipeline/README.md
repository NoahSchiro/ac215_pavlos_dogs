# Vertex AI training pipeline

Runs the existing training container on Vertex AI as a four-step pipeline:

```
preprocess  ->  train  ->  evaluate  ->  publish
(build cache)   (train.py)  (metrics)     (model for the UI)
```

The last step is the hand-off to serving: it copies the run's `best.onnx` to
`gs://<models-bucket>/models/<run>/gazenet.onnx` and outputs its public URL,
which is the `MODEL_URL` the UI is deployed with (`deploy/README.md`).

Every step runs the **same image** that `model/Dockerfile` already builds, with a
different command. No training logic lives here. A change to `train.py` needs no
change in this directory -- the pipeline decides only *what* runs, *in what
order*, and *on what machine*.

## Why a pipeline rather than `docker run`

Local training was bounded by the laptop: 16 GB of RAM, no CUDA, and an epoch
that slowed from 705 s to 1973 s once the dataloader workers pushed the machine
into swap. Vertex runs the same container on a machine sized for the job, keeps
the dataset on the bucket instead of the disk, and records each run so results
are attributable to an image digest and a parameter set rather than to whatever
was checked out that afternoon.

The three steps are split because they fail differently. Preprocessing is a long
one-time MediaPipe pass; training is the expensive part worth retrying; the
metrics step is cheap and must run even when you only want the number.

## Layout

| File | Purpose |
| --- | --- |
| `gaze_pipeline.py` | the DAG: three `dsl.container_component`s plus the `@dsl.pipeline` |
| `submit.py` | compiles the spec, optionally submits it to Vertex |
| `tests/test_pipeline.py` | compiles the pipeline and asserts the DAG, image and flags |

## Storage

Vertex mounts Cloud Storage under `/gcs`, so `gs://bucket/path` is an ordinary
directory inside a step at `/gcs/bucket/path`. That is how the dataset reaches
preprocessing and how checkpoints outlive the run. `gcs_mount()` does that
rewrite; it rejects anything that is not a `gs://` URI rather than silently
producing a path that would not exist.

Steps are ordered with `.after()` rather than by passing artifacts between them.
The cache and the checkpoints already live in the bucket, so the only thing the
DAG needs to express is sequence. (KFP container components also cannot take
`dsl.InputPath` parameters, so artifact passing was not an option here.)

## Running it

Compile only -- no GCP credentials needed, which is what the tests do:

```bash
uv run submit.py --compile-only
```

Build and push the training image, then submit:

```bash
# from the repository root
REGION=us-east1
PROJECT=<your-project-id>
IMAGE=$REGION-docker.pkg.dev/$PROJECT/gaze/gaze-model:latest

gcloud artifacts repositories create gaze \
    --repository-format=docker --location=$REGION   # first time only
gcloud auth configure-docker $REGION-docker.pkg.dev

docker build --platform linux/amd64 -t $IMAGE model/
docker push $IMAGE

cd pipeline
uv run submit.py --project $PROJECT --image $IMAGE --epochs 24
```

**W&B on Vertex.** Training logs to the team's W&B project (`gaze`, under
`noahschiro-harvard-university`). A Vertex container can't `wandb login`, so
the train step reads the API key from the private data bucket at runtime:

```bash
# once; the key must belong to a member of the W&B team
printf %s "<your W&B API key>" > /tmp/wandb_api_key
gcloud storage cp /tmp/wandb_api_key gs://ac215-gaze-data-rh/secrets/wandb_api_key
rm /tmp/wandb_api_key
```

The key never becomes a pipeline parameter, because parameters are stored
in the run's spec and shown in the Vertex console. If the key is missing, the
step fails before training starts. Pass `--wandb-project ''` to train without
W&B.

The model is published to `--models-bucket`, which defaults to
`<project>-gaze-models`, the public bucket `deploy/setup-gcp.sh` creates. The
URL appears as the `model_url` output of the publish step in the run graph and
in its logs (`published MODEL_URL=...`). Deploy it with:

```bash
MODEL_URL=<that url> deploy/deploy-ui.sh
```

A published version is never overwritten with different weights.
Republishing identical bytes succeeds, so a retried step doesn't fail on its
own earlier output. Objects written through the `/gcs` mount get Cloud
Storage's default one-hour cache; `deploy/publish-model.sh` sets a longer
`immutable` one.

`--image` must be set before the pipeline module is imported, because KFP bakes
the image into the compiled spec at compile time; it is deployment
configuration, not a pipeline parameter. `submit.py` handles that ordering and
refuses an image that still contains the `PROJECT` placeholder.

Other flags: `--region`, `--bucket`, `--val`, `--test`, `--wandb-project`.

`enable_caching=True` means a rerun with unchanged parameters reuses the
completed preprocess step instead of repeating the MediaPipe pass.

## Monitoring a run

Four places, in increasing detail:

1. **Vertex run graph** (`console.cloud.google.com/vertex-ai/pipelines`) -- step
   status, wall-clock per step, and the `metrics` artifact that the evaluate
   step publishes from `best_metrics.json`.
2. **Cloud Logging** -- every step's stdout/stderr is collected automatically;
   no logging code is needed in the container. Filter one run with:
   ```bash
   gcloud logging read \
     'resource.type="aiplatform.googleapis.com/PipelineJob"' \
     --project $PROJECT --limit 100 --format='value(textPayload)'
   ```
   This is where a crash inside MediaPipe or a corrupt shard shows up.
3. **Weights & Biases** -- per-epoch train/val loss and angular error, from
   `train.py`'s existing `--wandb` integration. Pass `--wandb-project`.
4. **The bucket** -- `checkpoints/<timestamp>/` holds `best.pt`, `last.pt`,
   `best_metrics.json` and the run log for every run, so a run is reproducible
   after the fact.

## Resource requests

DataLoader workers pass batches through `/dev/shm`, which the pipeline can't
size: Docker's default is 64 MB, and running out kills training mid-epoch
with a misleading "No space left on device". The train step measures
`/dev/shm` first. Below 1 GB it runs with `--workers 0`, loading data in the
main process: slower, but needing no shared memory. The decision is the
first line of the step's log. (`set_sharing_strategy("file_system")` doesn't
help here; it changes how segments are named, not where they live.)

Training asks for 32 GB and 8 CPU. The dataloader passes 224x224 crops between
worker processes through shared memory, which is what made the local runs thrash;
requesting the memory explicitly is what keeps that from recurring on Vertex.

No GPU is requested. The project's free-trial quota does not permit one, and
`train.py --device auto` selects CUDA when an accelerator is present, so adding
`.set_accelerator_type(...)` to the train step is the only change needed once
quota is granted.

## Development

```bash
cd pipeline
uv sync
uv run ruff format . && uv run ruff check . && uv run mypy . && uv run pytest -q
```

Pinned to Python 3.12 (`.python-version`): KFP 2.18 does not install on 3.14,
which is what the training image uses. The two environments are independent --
the pipeline only names the training image, it never imports from `model/`.
