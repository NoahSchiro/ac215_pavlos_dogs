# AC215 Project: Gaze Tracking Maze

This repo tracks the progress of the final project for AC215 @ Harvard for Fall 2026.

## Milestone 3: what's new

Since Milestone 2, the model is trained in the cloud and the app is deployed and monitored:

```
Vertex AI pipeline ──▶ publish model to Cloud Storage ──▶ Cloud Run serves the UI ──▶ browser runs the model
(preprocess → train → evaluate → publish, logged to W&B)                 │
                                                       Cloud Logging + Monitoring ◀── usage telemetry
```

| Rubric item | Where | Docs |
| --- | --- | --- |
| Experiment tracking (W&B) | `model/` (`WandbLogger`), used by the Vertex train step | [model/README.md](./model/README.md), [pipeline/README.md](./pipeline/README.md) |
| Serverless training + ML workflow (Vertex AI Pipelines) | `pipeline/`: preprocess → train → evaluate → publish | [pipeline/README.md](./pipeline/README.md) |
| Serverless deployment (Cloud Run) | `deploy/`, `ui/Dockerfile`, `ui/nginx/` | [deploy/README.md](./deploy/README.md) |
| Production monitoring | Browser telemetry → Cloud Logging → log-based metrics + alert (`deploy/monitoring/`) | [deploy/README.md#production-monitoring](./deploy/README.md#production-monitoring) |
| Prediction API | Not applicable: see below | |

**Why there is no prediction API.** Gaze inference runs in each user's browser
(onnxruntime-web), so webcam frames never leave the device and there's no
network round trip per prediction. A server-side prediction API would add
latency and a privacy risk without adding anything, so the backend instead
serves the app and the published model and collects monitoring data. This was
discussed with and approved by our TF and Pavlos.

### Quick links

- **Reproduce a training run on Vertex AI:** [pipeline/README.md → Run it](./pipeline/README.md#run-it)
  (`uv run submit.py --project $PROJECT --image $IMAGE --epochs 2`)
- **Deploy or redeploy the app:** [deploy/README.md → Deploy](./deploy/README.md#deploy)
  (`MODEL_URL=... deploy/deploy-ui.sh`)
- **Live app:** `<add Cloud Run URL here>`
- **Logs and dashboards:** [deploy/README.md → Production monitoring](./deploy/README.md#production-monitoring)
- **Evidence (screenshots):** [docs/](./docs/): W&B training run, Vertex pipeline run, `curl` against the deployed service

Other additions since Milestone 2: per-user ridge calibration in the UI, Kalman
smoothing and a gaze crosshair, and a shared UI config file (`ui/src/config.ts`).

## Proposal

`./proposal/`

Full proposal can be found [here](./proposal/README.md). The basic idea is that users will use [gaze tracking](https://en.wikipedia.org/wiki/Eye_tracking) to guide a ball through a maze.

## Training and backend

`./model/`

Training and backend deployment (API wrapper, containerization, `kube.yaml`)

### Data

`./model/data/` (.gitignored)

While I would include the data here through `git lfs`,

- The GazeCapture dataset comes with a license which does not allow for redistribution. It comes as a `tar` full of `tar.gz` so to unpack:
  - Move the `tar` to `./data/`
  - `tar -xf ./data/GazeCapture.tar -C ./data/GazeCapture/`
  - `cd data/GazeCapture/`
  - `for file in *.tar.gz; do tar -xzf "$file" && rm "$file"; done`
- The MPIIFaceGaze dataset has no license restrictions, and as such, can be fetched by `git clone`-ing this repo (**if you have `git lfs` installed!!**). Once you have cloned, just run `tar -xzf data/MPIIFaceGaze.tar.gz -C data`

### Running training

Once you are in the `./model/` directory (and you have unpacked data), running training takes one command:

```bash
uv run train.py
```

To see what options exist, run: `uv run train.py --help`

## Web UI

`./ui/`

The Gaze Maze frontend is built with React, TypeScript, and Vite. The UI is containerized using a multi-stage Docker build: Node.js builds the application, and Nginx serves the compiled files.

## Running everything with Docker Compose

From the project root, run:

```bash
docker compose up
```

Open the application in your browser:

**http://localhost:8080**

To verify that the container is running:

```bash
docker compose ps
```

To stop:

```bash
docker compose down
```

If you want to just run the UI or training:
```bash
docker compose up ui
docker compose up train
```

