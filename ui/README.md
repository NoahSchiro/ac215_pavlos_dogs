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
  calibration/
    points.ts         calibration/validation targets, sample types, burst constants
    capture.ts        video frame -> JPEG blob, burst capture
    useCamera.ts      owns the webcam MediaStream
    Calibration.tsx   the calibration screen
    AccuracyCheck.tsx  temporary four-point accuracy check
    validation.ts     paired raw/corrected mean error on fresh frames
    fullscreen.ts     best-effort enter/exit full screen
    Crosshair.tsx     target marker
    ridge.ts         pure centered ridge fit and affine application
    process.ts       sequential inference, coverage/variation checks, fitting
    record.ts        versioned calibration metadata and validation
    storage.ts       atomic IndexedDB save, restore, and clear
    cameraMetadata.ts  capture identity and camera compatibility checks
  gaze/
    GazeCrosshair.tsx  overlay for the displayed gameplay gaze
    kalman.ts         Kalman smoothing of per-frame gaze predictions
  vision/             the model's image preprocessing, ported to the browser
    cv.ts             thin wrappers over @techstark/opencv-js (see below)
    utils.ts          port of model/src/data/utils.py, steps 2-5 + tensors
    model.ts          GazeNet inference via onnxruntime-web (see below)
    index.ts          public surface
    utils.test.ts     line-for-line port of model/tests/test_utils.py
    parity.test.ts    parity against recorded native cv2 outputs
    model.test.ts     inference parity against Python onnxruntime
    fixtures/         face.jpg + reference outputs generated in model/
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

## Gaze inference

`src/vision/model.ts` runs the trained GazeNet in the browser with
`onnxruntime-web`. The model file is `public/models/gazenet.onnx`, exported
by the training pipeline's own `export_onnx` (`model/src/training/export.py`)
from the best checkpoint's `best.pt`; copy it here after training:

```bash
cp model/checkpoints/<run>/best.onnx ui/public/models/gazenet.onnx
```

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

Restore the actual ONNX weights with Git LFS:
`git lfs pull --include="ui/public/models/gazenet.onnx" --exclude=""`.
When replacing weights, update `CALIBRATION_COMPATIBILITY.modelId` in
`src/calibration/record.ts`. Update the preprocessing or intrinsics version
when those contracts change, so old fits are rejected.

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

Gameplay applies a matching personal ridge correction before Kalman smoothing.
The crosshair and movement use the same smoothed point. Calibration fitting and
the four-point accuracy check continue to use raw model predictions.

## Calibration

Clicking **Calibration** on the home screen opens the webcam first, while the
page is still windowed, so the browser's permission prompt stays visible.
(Safari attaches the prompt to the toolbar, which full screen hides.) Once the
camera is ready a **Camera ready** screen appears; clicking **Begin** (or
pressing Space) enters full screen and shows the first crosshair. Targets sit
on a 3x3 grid, inset 10% from each edge, ordered left to right then top to
bottom.

- **Space** captures a burst of 3 frames 100 ms apart, labels each with the
  current target, and advances to the next crosshair.
- **Escape** cancels and returns home.
- After the 9th target, capture stops and the home screen shows inference
  progress. **Calibration ready** appears after saving the ridge fit
  (`lambda = 0.1`). All valid pairs are fitted; every target needs a valid frame.
- Target-average predictions must have a standard deviation of at least
  `0.01` along the least-varying direction. Failed frames retain their reasons.
  Missing coverage or variation offers **Retry calibration**. Failed or
  cancelled processing and failed writes preserve the previous calibration.
- If the camera is denied or missing, an error screen offers **Retry** and
  **Back**.

Each `CalibrationSample` holds the full webcam frame as a JPEG `Blob`, the
target as normalized `(x, y)` in `[0, 1]` with `(0, 0)` at the top-left, the
target index, the viewport size, and a timestamp. This is the same label space
the gaze model predicts in. JPEGs, prediction/drop outcomes, frame
counts, fit parameters, lambda, and model/camera/capture metadata are stored
locally in IndexedDB. They are never uploaded.

Full screen is best effort: if the browser refuses, calibration runs windowed.

## Restore, reset, and reuse

- Reload restores compatible records from the same browser origin without
  opening the camera. Select **Verify saved camera** to compare device,
  facing mode, and image dimensions; unknown or changed settings need recalibration.
- **Recalibrate** saves a replacement. **Clear calibration** deletes the
  JPEGs and fit. Failed writes or clears preserve the previous record;
  corrupt or incompatible records are not activated.
- Recalibrate after changing posture, screen/window placement, or lighting.
  Matching metadata cannot verify those conditions. Calibration and gameplay
  share the same approximate camera intrinsics.

## Gameplay

Open **Maze** to start live prediction. A verified fit corrects each prediction
once when the camera and frame dimensions match. Otherwise, gameplay skips the
personal correction and reports when verification or recalibration is needed.
Both paths pass through Kalman smoothing before updating the crosshair and movement.
The direction boundary rejects non-finite values and clips corrected coordinates
to `[0, 1]`. Missing faces and prediction failures stop gaze movement.
Arrow keys remain available, including when camera access fails. The existing
direction thresholds, movement timing, and collision rules are unchanged.

## Accuracy check

After calibration (or verifying a restored camera), select **Validate
calibration**. The check reuses the capture screen with four targets, each
collecting three fresh frames with **Space**. It compares raw and corrected
mean Euclidean error against the same targets using the same valid frames.
Errors use normalized screen coordinates without clipping; lower is better.
Every target needs at least one valid frame.

The result shows both means and the valid frame count. **Try again** captures
another check; **Back to Home** cancels processing or closes the result. The
check does not refit or save calibration, and results only last on this page.
Automated tests do not establish a real-user accuracy improvement.

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
