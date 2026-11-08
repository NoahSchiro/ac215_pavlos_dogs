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

For the CI-compatible component, algorithm, workflow, and storage tests:

```bash
npm test -- --project app
```

`npm test` runs both the app and real-browser vision projects. The vision
project requires a desktop browser, runtime downloads, and the actual ONNX
weights.

## Lint

```bash
npm run lint
```

The webcam only works on `localhost` or over HTTPS. Current Chrome and Safari
are the supported browsers.

## Structure

```
src/
  App.tsx             home, maze, calibration, and independent validation
  home/               HomeScreen: choose maze or calibration
  maze/               live gaze loop, corrected direction mapping, maze controls
  calibration/
    points.ts         the 9 targets, sample types, burst constants
    capture.ts        video frame -> JPEG blob, burst capture
    useCamera.ts      owns the webcam MediaStream
    Calibration.tsx   the calibration screen
    fullscreen.ts     best-effort enter/exit full screen
    Crosshair.tsx     target marker
    ridge.ts         pure centered ridge fit and affine application
    process.ts       sequential sample inference, coverage checks, fitting
    record.ts        versioned calibration metadata and validation
    storage.ts       atomic IndexedDB save, restore, and clear
    cameraMetadata.ts  capture identity and camera compatibility checks
    validation.ts    held-out capture protocol and paired error measurement
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

The committed ONNX file uses Git LFS. It must contain the actual weights,
not the small text pointer. On a checkout with Git LFS installed, restore it
with `git lfs pull --include="ui/public/models/gazenet.onnx" --exclude=""`.
The current file is 46,084,415 bytes with SHA-256
`c4a11ab96dc64766cd1abd622838b50b297aaec4adf4f25ebf71cf0b66bcbee6`.
If replacing the model, update `CALIBRATION_COMPATIBILITY.modelId` in
`src/calibration/record.ts` to identify the new weights. Bump the preprocessing
or intrinsics version when those contracts change, so old fits are rejected.

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
- After the 9th target the camera stops and full screen exits. The home
  screen shows inference progress, then saves a ridge correction with
  `lambda = 0.1`. **Calibration ready** appears after the save commits.
- Failed frames retain their drop reasons. Fitting requires at least one
  valid frame at each of the nine targets and uses all valid pairs. Missing
  coverage offers **Retry calibration**. Cancelled or failed processing and
  failed writes preserve the previous saved calibration.
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

- Saved records belong to the same browser origin, including protocol and
  port. Reload restores a compatible record without opening the camera.
- Select **Verify saved camera** after restoration. This briefly opens the
  capture camera, compares device identity, facing mode, and actual image
  dimensions, then stops the camera. Unknown or changed settings require
  recalibration. A restored fit stays pending until verification succeeds.
- **Recalibrate** captures and saves a replacement. **Clear calibration**
  removes the saved JPEGs and fit; a failed clear retains the existing record.
- Corrupt records, incompatible versions, and known metadata mismatches do
  not become usable calibrations. A failed save reports that the new fit was
  not saved and keeps the prior record.
- Matching camera settings cannot establish unchanged posture, screen/window
  placement, or lighting. Recalibrate after those conditions change. Camera
  intrinsics are approximations, shared between capture and gameplay.

## Gameplay

Open **Maze** to start live webcam prediction. A verified fit is applied once
after each successful raw prediction when the live camera identity and frame
dimensions match its metadata. The direction boundary rejects non-finite
coordinates and clamps finite corrected coordinates to `[0, 1]`.

Without a usable calibration, the maze uses raw predictions and reports when
verification or recalibration is needed. Missing faces, rejected frames, and
prediction failures stop gaze movement. Arrow-key controls remain available,
including when the camera is denied or unavailable.

This branch incorporates the live loop from PR #68 at commit
`2d1bb40eebf46406f20065425622af092b19c4c5`, selected for integration before
that PR merges. Its center thresholds (`0.4`/`0.6`), prediction delay
(`200 ms` after completion), movement interval (`160 ms`), and collision
rules are preserved.

## Independent accuracy check

1. Complete a new calibration, or restore one and select **Verify saved
   camera**. Keep your position and camera setup fixed.
2. Select **Validate calibration**. This captures three new frames at each
   of four fixed targets, in order: `(0.25, 0.25)`, `(0.75, 0.25)`,
   `(0.25, 0.75)`, `(0.75, 0.75)`. Look at each target and press Space.
3. Read the raw and corrected mean errors and valid-frame count. Each target
   needs at least one valid paired frame. Both means use exactly the same
   accepted frames and their own target labels.
4. Copy **Validation metrics JSON** into the execution record for review.
   It contains the fixed targets, errors, counts, fit/model/camera metadata,
   and per-frame predictions/failures. It contains no JPEG data. The report
   is held in memory and is cleared when a new calibration starts or the
   current calibration is cleared.

The metric is mean Euclidean distance in normalized screen coordinates;
lower is better. The primary corrected metric uses the **unclipped** ridge
output; gameplay clipping is separate. Validation never refits the model,
changes lambda, or replaces the saved calibration. Known fitting-frame
reuse and captures predating the fit are rejected.

If corrected error does not improve, investigate orientation, camera
assumptions, posture, image quality, and regularization before claiming an
accuracy gain. Do not change the validation targets after seeing results
or report fitting error as independent accuracy evidence. Synthetic tests
prove arithmetic and integration, not improvement for a real player.

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
