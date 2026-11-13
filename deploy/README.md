# Deploying the app: Cloud Run, a published model, production monitoring

```
 Vertex AI pipeline  ──publish──▶  gs://MODELS_BUCKET/models/<version>/gazenet.onnx   (public, CORS)
 (or publish-model.sh)                         ▲
                                               │ browser fetches the model directly
 Cloud Run service "gaze-maze" ──serves UI──▶ browser ── runs the model on the webcam, locally
   nginx, MODEL_URL=<that URL>                 │
        ▲                                      │ aggregate telemetry beacons (/t)
        └──────── stdout JSON ◀────────────────┘
                     │
            Cloud Logging ──▶ log-based metrics ──▶ Cloud Monitoring (charts, alert)
```

Inference happens in the user's browser, so the server only serves static
files. That decides most of what follows:

- **Cloud Run *service*, not a Cloud Run *function*.** The app is a container
  serving files (nginx), which is exactly what a service runs. Functions
  deploy request-handling code from source. There is no request handler here,
  and one would put an inference round trip where none is needed.
- **The model is runtime configuration.** One UI image serves any model: the
  browser reads `MODEL_URL` from `/config.json`, which nginx renders from the
  container's environment at startup. Promoting a newly trained model is a
  redeploy with a different `MODEL_URL`, not a rebuild, and rolling back is
  switching to the previous revision.
- **Monitoring has to come from the browser.** Cloud Run's request logs show
  page loads, never whether gaze tracking works. The UI beacons per-session
  aggregates, which become metrics (below).

Region is `us-east1` throughout, matching `pipeline/submit.py`. Keep Artifact
Registry, Vertex AI, the buckets and Cloud Run together.

## One-time setup

Run these from [Cloud Shell](https://shell.cloud.google.com) (it has `gcloud`
and Docker) or anywhere with the gcloud CLI.

```bash
export PROJECT=<your-project-id>
deploy/setup-gcp.sh
export MODELS_BUCKET=$PROJECT-gaze-models REGION=us-east1
```

This creates the Artifact Registry repo, and a **public-read** bucket for
published models with CORS so browsers may fetch from it. It also gives
Vertex AI's service account write access to that bucket.

Only models go in that bucket. **Never make the data bucket public:**
GazeCapture's license forbids redistributing the dataset.

## Publish a model

Run cloud training with [`model/deploy.sh`](../model/deploy.sh). The Cloud Run
Job writes its training outputs to the results bucket. Download the completed
run directory containing `best.onnx`, or use the directory from a local run.

From the repository root, publish the model to your configured models bucket:

```bash
deploy/publish-model.sh /path/to/completed-run
```

Use the `MODEL_URL` printed by the script in the deployment command below.

## Deploy

```bash
MODEL_URL=https://storage.googleapis.com/... deploy/deploy-ui.sh
```

This builds the image with Cloud Build and deploys it as a new revision.
Cloud Build is used rather than a local `docker build` because Cloud Run
requires linux/amd64, and an Apple Silicon Mac builds arm64 by default.

Then check the live deployment the way a user would:

```bash
deploy/smoke-test.sh https://gaze-maze-<hash>-ue.a.run.app
```

To roll back, list the revisions with `gcloud run revisions list --service
gaze-maze` and pin traffic to an earlier one:

```bash
gcloud run services update-traffic gaze-maze --to-revisions <revision>=100
```

## Smoke test

`deploy/smoke-test.sh` drives the production UI in headless Chrome, with the
repo's MPII test face as the webcam. It passes once the maze reaches
"Tracking gaze", keeps it, and the model loaded. It needs the maze's gaze
loop (PR #68) to be present. With no argument it builds
`ui/` and tests that image; with a URL it tests a deployment. It needs only
Docker.

It exists because nothing else exercises what users run. The vision tests use
the Vite dev server, and CI skips them entirely. A production-only bug (OpenCV
failing to load in the built bundle, every frame `pose_failed`) shipped
through all of them. Against that build this test fails; against the fix it
passes.

## Production monitoring

```bash
deploy/setup-monitoring.sh                      # metrics, plus an alert with no recipient
ALERT_CHANNEL=projects/$PROJECT/notificationChannels/<id> deploy/setup-monitoring.sh
```

**What leaves the browser.** Counts, rates and timings per session, under a
random per-page id. No frames, no landmarks, no gaze coordinates.
`ui/src/telemetry.test.ts` asserts that.

| Beacon `ev` | Sent | Fields |
| --- | --- | --- |
| `start` | the model is requested (a gaze session starts) | `model`, `app` |
| `frames` | every 30 s, and when the page is hidden or closed | frame counts by outcome (`ok`, `face_not_detected`, `pose_failed`, …), `track_rate`, `lat_mean`/`lat_max` ms |
| `load_error` | model or face landmarker failed to load | `model` |

The numbers come from the vision pipeline's own events (`ui/src/vision/events.ts`):
`loadGazeModel`, `getLandmarker` and `predictGaze` report them, so every
caller is measured the same way and no screen needs instrumenting.

nginx validates every numeric field before logging. A malformed beacon can't
break or forge a log line: it is logged with zeros.

**Metrics** (`logging.googleapis.com/user/…`), all split by model version:

| Metric | Answers |
| --- | --- |
| `gaze_track_rate` | Is tracking working for people? It falls when faces stop being found or a model regresses. |
| `gaze_inference_latency` | How responsive is control? It's per-frame browser time, preprocessing plus ONNX. |
| `gaze_model_load_errors` | Can browsers load the model at all? **Alerted on.** |
| `gaze_sessions` | Usage, and the denominator for the error rate. |

In Metrics Explorer, chart `gaze_track_rate` with the 50th percentile, grouped
by `model`, to compare a new model against the old one in production.
Cloud Run's own metrics (request count, latency, instance count) need no setup.

## Running locally

Unchanged: `docker compose up` serves the UI on http://localhost:8080 with
the model bundled in the image (`git lfs pull` first). The same image honours
`MODEL_URL`, `TELEMETRY` and `PORT` if you set them. Telemetry lines print to
`docker compose logs ui`.
