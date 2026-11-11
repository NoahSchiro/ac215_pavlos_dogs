# Vertex AI training pipeline

```
preprocess  ->  train  ->  evaluate  ->  publish
(build cache)   (train.py)  (metrics)     (model for the UI)
```

Every step runs the image `model/Dockerfile` builds, with a different command.
No training logic lives here. `publish` copies the run's `best.onnx` to
`gs://<models-bucket>/models/<run>/gazenet.onnx` and outputs its public URL,
which is the `MODEL_URL` the UI deploys with (`deploy/README.md`).

## Run it

From [Cloud Shell](https://shell.cloud.google.com). Everything is pinned to
one commit: the image is built from that commit's `model/` and tagged with its
hash, so a run names the exact code that produced it.

```bash
export PROJECT=$(gcloud config get-value project) REGION=us-east1

COMMIT=2f7e379   # the commit to run; any commit that has pipeline/
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/NoahSchiro/ac215_pavlos_dogs.git
cd ac215_pavlos_dogs && git checkout $COMMIT
export IMAGE=$REGION-docker.pkg.dev/$PROJECT/gaze/gaze-model:$(git rev-parse --short HEAD)

gcloud builds submit model --tag $IMAGE --timeout 3600 --machine-type e2-highcpu-8

# Don't submit until the image exists: a run with a missing image fails late.
gcloud artifacts docker images list $REGION-docker.pkg.dev/$PROJECT/gaze --include-tags | grep $(git rev-parse --short HEAD)

curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env
cd pipeline && uv run submit.py --project $PROJECT --image $IMAGE --epochs 2
```

- Cloud Build produces linux/amd64, which Vertex requires. The longer timeout
  is because installing PyTorch can outlast the 10-minute default.
- `GIT_LFS_SKIP_SMUDGE=1` skips LFS files the build doesn't need.
- `--epochs 2` demonstrates a logged, reproducible run on CPU. Raise it for a
  model worth deploying.

**Before the first run:**
- Set up the project once. On new projects the Compute Engine default
  service account, which runs **both** Cloud Build and every pipeline step,
  starts with **no roles**. Each missing one fails a different stage (see
  [Troubleshooting](#troubleshooting)):
  ```bash
  export PROJECT=$(gcloud config get-value project) REGION=us-east1
  SA=$(gcloud projects describe $PROJECT --format 'value(projectNumber)')-compute@developer.gserviceaccount.com

  gcloud services enable aiplatform.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com
  gcloud artifacts repositories create gaze --location $REGION --repository-format docker
  gcloud storage buckets create gs://$PROJECT-gaze-models --location $REGION

  for ROLE in aiplatform.user cloudbuild.builds.builder logging.logWriter \
              artifactregistry.writer storage.objectAdmin; do
    gcloud projects add-iam-policy-binding $PROJECT --member serviceAccount:$SA \
      --role roles/$ROLE --condition None >/dev/null
  done
  # The data bucket may belong to another project; grant it there explicitly.
  gcloud storage buckets add-iam-policy-binding gs://ac215-gaze-data-rh \
    --member serviceAccount:$SA --role roles/storage.objectAdmin
  ```
- Put the MPII cache at `gs://<bucket>/MPIIFaceGaze/MPIIFaceGaze_wd/`, and
  preprocessing is skipped.
- Upload your W&B key. The train step reads it from the private bucket at
  runtime, because a Vertex container can't `wandb login`:
  ```bash
  read -rsp "W&B API key: " WANDB_KEY && echo     # hidden; never touches disk
  printf %s "$WANDB_KEY" | gcloud storage cp - gs://ac215-gaze-data-rh/secrets/wandb_api_key
  unset WANDB_KEY
  ```

## Flags

| Flag | Default | Purpose |
| --- | --- | --- |
| `--project`, `--region` | , `us-east1` | where to run |
| `--image` | | training image; baked into the spec at compile time |
| `--bucket` | `gs://ac215-gaze-data-rh` | data, checkpoints, pipeline root |
| `--models-bucket` | `<project>-gaze-models` | public bucket `publish` writes to |
| `--epochs`, `--val`, `--test` | `10`, `13`, `14` | passed to `train.py` |
| `--run-id` | new UTC timestamp | this run's folder under `checkpoints/`; reuse one to replay that run |
| `--wandb-project` | `gaze` | `''` trains without W&B |
| `--wandb-key` | `secrets/wandb_api_key` | the submitter's own key, in `--bucket` |
| `--no-cache` | off | rerun every step (see below) |
| `--compile-only` | off | write `gaze_pipeline.json`; no credentials needed |

Every submission gets its own run id (a UTC timestamp, printed by `submit.py`),
so it trains fresh into its own folder, while `preprocess` is still reused from
cache. Vertex caches steps by image and inputs, so passing an earlier
`--run-id` with unchanged parameters **replays** that run instead of repeating
it. Pass `--no-cache` to force every step to rerun.

## Checking a run

`submit.py` prints the job name (`gaze-training-<timestamp>`). The run's
state, each step's state, and the run's error if it has one:

```bash
JOB=gaze-training-20261111182741   # replace with the name submit.py printed
curl -s -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  "https://us-east1-aiplatform.googleapis.com/v1/projects/$PROJECT/locations/us-east1/pipelineJobs/$JOB" \
  | python3 -c "
import json,sys; j=json.load(sys.stdin); print('run:', j['state']); print('error:', j.get('error'))
for t in j.get('jobDetail',{}).get('taskDetails',[]): print(f\"  {t['taskName']:<14} {t.get('state','PENDING')}\")"
```

A step's live logs: `gcloud ai custom-jobs list --region us-east1`, then
`gcloud ai custom-jobs stream-logs <id> --region us-east1`.

## Troubleshooting

Each of these happened on our first run.

| Symptom | Cause | Fix |
| --- | --- | --- |
| `gcloud builds submit` → `FAILURE`, plus "does not have permission to write logs" | The service account has no Cloud Build roles, and without log access the real error is hidden | Grant `cloudbuild.builds.builder`, `logging.logWriter`, `artifactregistry.writer` (setup above), then rebuild |
| Run `FAILED` with **no steps listed**; error `Permission 'aiplatform.metadataStores.get' denied` | The service account can't use Vertex AI | Grant `roles/aiplatform.user`, wait a minute, resubmit |
| `submit.py` prints a traceback: "exists but does not belong to project … bucket squatting" | The data bucket is in another project | Harmless: the job is still created. Check for `PipelineJob created` |
| A step fails with permission denied on `/gcs/…` | No access to the (cross-project) data bucket | The `buckets add-iam-policy-binding` line in setup |
| Run fails on the image, or never starts a step | The image tag doesn't exist (the build failed) | Check the image exists before submitting (above) |
| `publish` fails: `no best.onnx` | The image was built from code that predates the ONNX export | Build from an up-to-date commit, and pin it with `git checkout` |
| `train` fails: `no API key at …` | The W&B key isn't uploaded | Upload it (above), or pass `--wandb-project ''` |
| `train` fails on quota (`custom_model_training_cpus`) | The project's CPU quota is below the step's 8-CPU request | Request more quota, or lower `set_cpu_limit` in `gaze_pipeline.py` |

## Who can rerun it

Each person reruns it with **their own credentials**, never someone else's.

- **Teammates** need `roles/aiplatform.user` on the project,
  `roles/storage.objectAdmin` on the data bucket, and
  `roles/artifactregistry.reader`. Each uploads their own W&B key, for example
  to `secrets/wandb_api_key_<name>`, and passes `--wandb-key`, so the run logs
  under their name.
- **Anyone else** can run it in their own project, with their own copy of the
  data, since every location is a flag. The one exception: `WandbLogger`
  hard-codes the team's W&B entity, so pass `--wandb-project ''` or edit it.

## Where to look

| Place | What's there |
| --- | --- |
| Vertex run graph | step status and timings; `evaluate`'s `metrics` and `publish`'s `model_url` outputs |
| Cloud Logging | every step's stdout/stderr, e.g. a MediaPipe crash or a corrupt shard |
| W&B, project `gaze` | per-epoch loss and error curves |
| `gs://<bucket>/checkpoints/<run-id>/<timestamp>/` | `best.pt`, `best.onnx`, `last.pt`, `best_metrics.json` |

## Design notes

- **Storage:** Vertex mounts buckets at `/gcs`, so steps read and write
  ordinary paths. Steps are ordered with `.after()` rather than by passing
  artifacts, because the data already lives in the bucket.
- **Secrets:** the W&B key is read from a file, never passed as a parameter,
  because parameters are shown in the Vertex console. A missing key fails
  before training starts.
- **`/dev/shm`:** DataLoader workers pass batches through it, and containers
  can have as little as 64 MB. The train step measures it and below 1 GB uses
  `--workers 0`, logging the choice. Changing the sharing strategy wouldn't
  help, since both strategies allocate in `/dev/shm`.
- **One folder per run:** train writes to `checkpoints/<run-id>/`, and
  evaluate and publish only look there. Picking "the newest run in the bucket"
  instead would grab another run's model whenever two runs overlap.
- **Publishing:** a published version is never overwritten with different
  weights. Republishing identical bytes succeeds, so a retried step doesn't
  fail on its own earlier output.
- **No GPU:** the free-trial quota has none. `train.py --device auto` uses
  CUDA when present, so adding `.set_accelerator_type(...)` to the train step
  is the only change needed later.

## Development

```bash
cd pipeline && uv sync
uv run ruff format . && uv run ruff check . && uv run mypy . && uv run pytest -q
```

This uses Python 3.12, because KFP doesn't install on 3.14, the version the
training image uses. The pipeline only names that image; it never imports
`model/`.
