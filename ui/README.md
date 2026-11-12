# Web UI for the gaze tracking maze

React + TypeScript + Vite frontend.

Run the following commands from the `ui` directory.

## Install dependencies

```bash
cd ui
npm ci
```

## Run locally

```bash
npm run dev
```

Open the local URL printed in the terminal, usually
http://localhost:5173. Press Ctrl+C to stop the development server.

## Build

```bash
npm run build
```

This runs TypeScript checks and creates the production bundle in `dist/`.

## Test

```bash
npm run test
```

Use `npm test -- --project app` for CI-compatible tests. The full suite also
runs vision tests, which require a desktop browser, runtime downloads, and
the actual ONNX weights.

## Lint

```bash
npm run lint
```

The webcam only works on `localhost` or over HTTPS. Current Chrome and Safari
are the supported browsers.

## Structure

```
src/
  App.tsx             home, maze, and personal calibration
  home/               HomeScreen: choose maze or calibration
  maze/               live gaze loop, corrected direction mapping, maze controls
  calibration/        nine-point calibration, saved correction, four-point check
  gaze/
    GazeCrosshair.tsx  overlay for the displayed gameplay gaze
    kalman.ts         Kalman smoothing of per-frame gaze predictions
  vision/             the model's image preprocessing, ported to the browser
    camera.ts         camera settings, verification, and approximate intrinsics
    viewport.ts       current window dimensions and size comparison
    jpeg.ts           decode JPEG blobs into RGB pixels
    cv.ts             thin wrappers over @techstark/opencv-js (see below)
    utils.ts          port of model/src/data/utils.py, steps 2-5 + tensors
    model.ts          GazeNet inference via onnxruntime-web (see below)
    index.ts          public surface
    utils.test.ts     line-for-line port of model/tests/test_utils.py
    parity.test.ts    parity against recorded native cv2 outputs
    model.test.ts     inference parity against Python onnxruntime
    fixtures/         face.jpg + reference outputs; see note below
  test-setup.ts       jest-dom matchers + Testing Library cleanup
```

## Vision preprocessing

`src/vision/` is the browser counterpart of the model's image preprocessing,
so the ONNX export of GazeNet can run on artifacts identical to the ones
training used. The cv2 geometry is `@techstark/opencv-js` (pinned to the exact
version `opencv-python-headless` uses in `model/`) behind thin wrappers in
`cv.ts`. It takes an RGB uint8 image plus camera intrinsics and produces:

- the 224x224 normalized face crop,
- the 36x60 left/right eye crops (anatomical left/right),
- the virtual-camera rotation `R_norm`,
- `prepareImages` tensors: uint8 crops -> float32 (B, 3, H, W) with the
  ImageNet statistics the pretrained face tower expects.

`preprocess` runs the whole chain on one frame and quality-gates it, returning
the artifacts or a drop reason. The geometry is asynchronous because the
OpenCV WASM runtime loads once per page; opencv.js memory is not garbage
collected, so the wrappers delete every Mat on all exit paths.

The FaceLandmarker is `@mediapipe/tasks-vision` with the same options as
`get_landmarker` in `model/src/data/utils.py`. It fetches `landmarker.task`
from the same URL the Python side does, and its wasm runtime from the
jsdelivr CDN pinned to the installed package version.

### Parity fixtures

`fixtures/` holds `face.jpg` and the reference outputs `parity.test.ts`
compares this port against: the landmarks, head pose and homographies in
`ref.json`, the three expected crops as raw `.bin` bytes, and `cv2_probes.json`
-- recorded inputs and outputs for the individual cv2 primitives `cv.ts`
reimplements.

They are generated from the Python pipeline by
`model/tools/gen_parity_fixtures.py`, and CI regenerates and verifies them on
every pull request. Do not hand-edit them: if a test here starts failing
against a fixture, either this port drifted from `model/src/data/utils.py` or
the preprocessing changed and the fixtures need regenerating, and the
generator is what tells the two apart. See `model/README.md` for how to run it
(it needs Linux, and should be run on amd64 -- the crop comparisons here are
byte-exact and MediaPipe is not bit-reproducible across architectures).

## Gaze inference

`src/vision/model.ts` runs the trained GazeNet in the browser with
`onnxruntime-web`. The model file is `public/models/gazenet.onnx`, exported
by the training pipeline's own `export_onnx` (`model/src/training/export.py`)
from the best checkpoint's `best.pt`. Replace `RUN_NAME` with the training run
directory, then copy the model:

```bash
cp ../model/checkpoints/RUN_NAME/best.onnx public/models/gazenet.onnx
```

That bundled copy is only the default. Which model the browser loads is
runtime configuration (`MODEL_URL`, served as `/config.json` by the nginx
container), so a deployment points at a published model in Cloud Storage
without rebuilding the image: see `deploy/README.md`.

The public API mirrors the preprocessing pipeline:

- `loadGazeModel()` creates the inference session on the wasm CPU backend
  (once per page; the wasm runtime is fetched from the CDN pinned to the
  installed `onnxruntime-web` version, like the other runtimes above).
- `predictScreenFraction(model, frame)` feeds one `PreprocessedFrame`'s
  `prepareImages` tensors through the network.
- `predictGaze(landmarker, model, image, intrinsics)` is the end-to-end
  entry point for webcam frames: preprocess (landmarks, head pose, quality
  gates, normalization) then inference, returning the gaze screen fraction
  `{x, y}` or a `DropReason` for frames that fail a quality check.

Predictions are sigmoid-activated fractions of the screen — the same label
space `transformLabel` records calibration targets in.

[Git LFS](https://github.com/git-lfs/git-lfs/blob/main/docs/man/git-lfs-faq.adoc) tracks the model file.
With Git LFS configured for automatic downloads, `git pull` also retrieves
updated weights. If the model file is still an LFS pointer, download the weights:

```bash
git lfs pull --include="ui/public/models/gazenet.onnx" --exclude=""
```

## Gaze smoothing

`src/gaze/kalman.ts` smooths the per-frame `predictGaze` output with a
constant-velocity Kalman filter (independent x/y axes, variable frame timing).
Feed it every frame, passing `null` when `predictGaze` returns a `DropReason`:

```ts
const kf = new GazeKalmanFilter()
const raw = await predictGaze(landmarker, model, image, intrinsics)
const gaze = kf.update(typeof raw === 'string' ? null : raw, performance.now())
```

A single far-off prediction is gated out as an outlier; two in a row are
treated as a saccade and the filter jumps to the new point. Tune
`measurementStd` (R) and `accelerationStd` (Q): higher R or lower Q gives
smoother but slower output.

## Calibration

Personal calibration fits a ridge correction to your gaze predictions.
Gameplay applies the correction before smoothing. The crosshair and maze
movement use the same corrected, smoothed prediction.

1. Select **Calibration** on the home screen.
2. Allow camera access and wait for **Camera ready**.
3. Select **Begin**, or press Space, to enter full screen.
4. Look at each of the nine targets and press Space to capture three frames.
5. Wait for **Calibration ready** on the home screen before playing.

Press **Escape** to cancel. If a capture fails, press Space again.
If processing fails, select **Retry calibration**.
If camera access fails, select **Retry** or **Back**.
Failed or cancelled processing preserves the previous calibration.

The browser decodes captured images into RGB pixels before preprocessing and inference.
Images stay in memory during processing and are never uploaded or saved.
The browser saves only the fitted correction and information needed to reuse it.

If the browser refuses full screen, keep the window size and position unchanged.
If the window size or full-screen mode changes during capture, go back and
restart calibration.

## Restore, reset, and reuse

- After a reload, select **Verify saved camera** before reusing a saved calibration.
  If verification fails, recalibrate with the current camera.
- **Recalibrate** replaces the saved correction. **Clear calibration** deletes it.
  Failed saves or clears preserve the previous record.
- Recalibrate after changing your camera, posture, window position, or lighting.
  If an update replaces the model weights, recalibrate before playing.

Updating from the earlier format that saved photos clears the old record.
Complete calibration again once after this update.

## Gameplay

1. Open **Maze** and allow camera access.
2. Wait for the camera, then select **Start game** to enter full screen and start tracking.

You can play with or without calibration. The status message shows whether
gameplay uses your correction. Arrow keys also control the maze.

If you use calibration, keep the game window at the size used for calibration.
Changing window size or full-screen mode pauses gaze control and hides the crosshair. Restore the original
size and mode, or recalibrate. Use **Restore full screen** to return to full screen.
Missing faces and prediction failures also pause gaze control.
Select **Back to Home** to leave the maze and exit full screen.

## Accuracy check

1. After calibration or camera verification, select **Validate calibration**.
2. Select **Begin** and use the same window size as calibration.
3. Look at each of the four targets and press Space to capture three fresh frames.
4. Compare **Raw mean error** with **Corrected mean error**. Lower error is better.

The check compares predictions before and after correction on the same fresh frames.
Errors use normalized screen coordinates. The check does not change your calibration.
Select **Try again** for another check, or **Back to Home** to leave.

## Testing

`npm run test` runs two Vitest projects:

- **app** — component tests with `@testing-library/react` on jsdom. jsdom
  has no camera, canvas, or full-screen API, so tests mock
  `navigator.mediaDevices`, `HTMLCanvasElement.prototype.toBlob`, and
  `document.requestFullscreen`. Testing Library's automatic cleanup is
  registered in `src/test-setup.ts` because Vitest runs without globals.
- **vision** — the preprocessing tests in `src/vision/`, run in a real
  browser through Vitest's `preview` provider, because MediaPipe's
  FaceLandmarker uploads frames through WebGL, which jsdom cannot provide.
  The provider opens your **system default browser** as a regular tab.
  Point it at a specific browser with the `BROWSER` env var (e.g.
  `BROWSER=firefox npm test`), and pass browser flags via `BROWSER_ARGS`.
  If no browser can be opened (e.g. a bare CI container), the vision tests
  error out.

The vision tests download the face landmarker model (~3.6 MB) on first use,
like the Python tests do. The browser tab stays open after the run.
