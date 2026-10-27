# Gaze Tracking model and deployment

## Data preprocessing

The preprocessing pipeline turns the raw [MPIIFaceGaze](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/its-written-all-over-your-face-full-face-appearance-based-gaze-estimation) dataset into a quality-filtered, normalized torch `Dataset`. Everything lives in `src/data/`

### Setup

```bash
uv sync
```

If you have `git cloned` with `git lfs` installed, your data will be placed in the correct location.

```
data/MPIIFaceGaze/
├── p00/
│   ├── p00.txt                  # one line per image, 28 whitespace-separated columns
│   ├── day01/0005.jpg           # face crops
│   └── Calibration/
│       ├── Camera.mat           # camera intrinsics (fx, fy, cx, cy)
│       └── screenSize.mat       # screen size in pixels and millimeters
├── p01/ ... p14/
```

The Face Landmarker model (`models/landmarker.task`) is downloaded automatically on first use. Note: MediaPipe's runtime needs `libEGL`/`libGLESv2` on Linux.

### Pipeline steps

**1. Parsing** — `parse_mpii()` / `parse_mpii_sample()` (`src/data/mpiifacegaze.py`)
The only code that knows the on-disk MPIIFaceGaze format. Of each annotation line's 28 columns we use the image path and the gaze target on screen (pixels). Per-subject calibration comes from `Camera.mat` (focal length = mean of fx/fy, principal point) and `screenSize.mat` (screen size in px and mm). Anything that doesn't parse is dropped with a printed reason and counted.
  * Output: `Sample` records per the schema in `src/data/schema.py`. This is a standard format every later step consumes. Every dataset must output a list of `Sample`

**2. Face/landmark detection** — `landmark_detection(landmarker, image)` (`src/data/utils.py`)
MediaPipe Face Landmarker (478 points with iris refinement) on RGB uint8 images. 
  * Output: pixel-space keypoints, or `None` when no face is found. The detector is the same one intended for the browser inference.

**3. Head pose** — `head_pose_est(landmarks, intrinsics)`
Fits a generic 3D face model to the landmarks with `cv2.solvePnP`. 
  * Output: rotation `R` (3×3) and translation `t` (centimeters) mapping the model into the camera frame; a face looking straight at the camera gives `R = np.eye(3)` (identity matrix).

**4. Normalization** — `normalize_face(image, landmarks, R, t, intrinsics)`
Warps the image so the face is upright, fronto-parallel and at a fixed scale, regardless of original pose/distance. We do this by placing a virtual camera 60 cm along the head's viewing axis (focal 960 px) and the image is resampled by mapping the true camera to this virtual camera. 
  * Output: 224×224 face crop, 36×60 left/right eye crops, and the virtual-camera rotation `R_virtual`.

**5. Label transform** — `transform_label(gaze_target_px, screen_size_px, R_virtual, gaze_direction_cam=None)`
Canonical label = normalized screen fraction `(x/W, y/H)` If a 3D gaze direction is available, it is also rotated by `R_virtual.T` into the normalized frame as a unit vector. Fractions are deliberately not clamped; out-of-bounds labels are filtered in the next step.
  * Output: normalized labels as fraction of screen and as 3D gaze direction

**6. Quality filtering** — `filter_dataset(samples, num_threads=None)`
Iterates through all `Sample` and drops samples if they cannot pass these preproceesing steps. Prints a summary on what was dropped and why.
  * Output: only high quality `Sample`

**7. torch Dataset** — `MPII(participants, dataset_root)` (`src/data/mpiifacegaze.py`)
Loads the filtered manifest, or builds it from scratch and places it at `data/MPIIFaceGaze/mpii_filtered.json`. Selects participants for subject-disjoint train/test splits. `__getitem__` loads the image and computes steps 2–5 **on the fly**, returning:

```python
{
    "sample_id": str,                  # "mpiifacegaze/p00/day01/0005"
    "face":       uint8 (224, 224, 3), # normalized face crop (RGB)
    "left_eye":   uint8 (36, 60, 3),   # subject's anatomical left eye
    "right_eye":  uint8 (36, 60, 3),   # subject's anatomical right eye
    "label":      float32 (2,),        # normalized screen fraction
    "R_norm":     float64 (3, 3),      # virtual-camera rotation
}
```

### Running it

The cache is built on first use, so the first run is slow and later ones are
not. `train.py` is the only entry point.

### Notes

- Crops and labels are computed once into webdataset shards, so `__getitem__` is a shard read plus a PNG decode rather than a MediaPipe pass. The shards are ~2.9GB for MPIIFaceGaze.
- Filtering thresholds live as constants in `filter_dataset`; the virtual-camera parameters are constants in `normalize_face`. Both must stay identical across datasets and match the TypeScript port of steps 2 and 4 planned for browser inference.
- Adding a second dataset should be simple write a new parser to `Sample`, pass those samples through `filter_dataset()` and wrap it up in a torch Dataset.

## Training

### Running it

```bash
uv run train.py --val 13 --test 14 --epochs 30
```

The crops are cached as a webdataset, so training reads from disk and does not
run MediaPipe. It uses CUDA, then Apple MPS, then CPU, whichever is available
first; override with `--device`.

`--dataset gazecapture` trains on GazeCapture instead. MPII splits by participant
via `--val/--test`; GazeCapture ignores those and uses its own official
subject-disjoint train/val/test splits.

`--weights PATH` initializes the model from a previous run's checkpoint
(weights only: fresh optimizer and epoch counter, so it lands in its own new
run directory). The checkpoint's architecture flags must match the current
invocation (`--output-mode`, `--no-pretrained`, `--no-eyes`, `--dropout`);
`load_checkpoint` fails loudly on a mismatch. `--resume` instead continues a
run in place and is mutually exclusive with `--weights`.

`--lr-schedule cosine` decays the learning rate toward zero across `--epochs`
instead of holding it constant (default `none`); the lr used each epoch is
logged as `lr`.

Augmentation is off by default and runs per batch on the GPU during training
only (evaluation always sees raw crops): `--flip` mirrors half the samples
(eye crops swap sides and labels transform to match), `--crop-jitter` adds
+/- 5° rotation, +/- 10% scale and +/- 4% translation to the crops, and
`--photometric-jitter` adds +/- 20% brightness/contrast. All three are recorded
in checkpoints.

After training, the test set is scored twice: raw, and with a per-user
ridge-regression correction that `train.py` fits itself on each test
subject's nine extreme screen positions (corners, edge midpoints and center 
of a 3x3 lattice. Both scores are logged

If you are using the `wandb` logger, then you need to login first:

```bash
uv run wandb login
uv run train.py --wandb gaze # project name is gaze
```

### Running it in a container

MediaPipe's FaceLandmarker aborts on some macOS setups inside Metal (observed
on an M-series Mac with both the CPU and GPU delegate), which blocks *building*
the cache. A Linux container has no Metal and falls back to XNNPACK:

```bash
docker compose run --rm train                      # defaults from docker-compose.yml
docker compose run --rm train --epochs 1 --val 5   # arguments override
```

Note Docker Desktop on macOS does not pass through the GPU, so a containerized
run is CPU-only. Build the cache in the container, then train natively to use
MPS.

`--shm-size=2g` is required when running `docker run` directly. PyTorch's
DataLoader passes tensors between workers through `/dev/shm`, which Docker
defaults to 64MB; a batch of 224x224 images overruns it and fails with
`No space left on device`, which looks like a full disk but is not. Compose
sets this via `shm_size`.

### Layout

| Module | Role |
|---|---|
| `src/training/model.py` | `GazeNet`: a ResNet-18 face tower plus one eye tower shared across both eyes, with the left crop flipped into the right's orientation. `--no-pretrained` switches to random init; `--no-eyes` ablates the eye towers. |
| `src/training/metrics.py` | Mean, median and p95, plus a per-participant breakdown, with converters to cm and degrees for comparison against published figures. |
| `src/training/loop.py` | Train and eval loops, per-epoch and best checkpointing, resume. |
| `src/training/logging.py` | `ConsoleLogger` (json logger), and a `WandbLogger` (logs to wandb) |

`--output-mode` selects the head. We train on `screen_fraction` today;
`gaze_direction` is wired up for the normalized-space formulation if we move
to it.

### Data versioning

`write_manifest` emits byte-identical JSON for identical input, so
`sha256 data/MPIIFaceGaze/mpii_filtered.json` is a dataset snapshot ID. Record
it alongside any reported result.
