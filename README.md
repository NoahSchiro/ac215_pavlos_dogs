# AC215 Project: Gaze Tracking Maze

This repo tracks the progress of the final project for AC215 @ Harvard for Fall 2026.

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

