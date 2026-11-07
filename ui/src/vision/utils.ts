/**
 * Primary preprocessing tools, ported line for line from
 * `model/src/data/utils.py` (these must stay identical in both languages):
 *
 * 1. Landmarker pulls out key features of face
 * 2. headPoseEst estimates head rotation/translation from those landmarks
 * 3. normalizeFace warps the image into a canonical face/eye frame
 * 4. transformLabel converts raw gaze labels to match the normalized image
 * 5. prepareImages turns uint8 crops into normalized float tensors
 * 6. preprocess runs every step on one frame and quality-gates the result
 *    (the browser counterpart of the Python quality filter; it takes the
 *    decoded image directly because the browser has no filesystem)
 */

import {
  getRectSubPix,
  invert3x3,
  matMul3x3,
  projectPoints,
  rodrigues,
  rodriguesToVector,
  solvePnP,
  warpPerspective,
  type Size2i,
} from './cv'
import type { FaceLandmarker } from '@mediapipe/tasks-vision'

// --- small numpy-style 3-vector helpers (kept as plain math: constructing
// Mats for three numbers would cost more than the operations) ---

/** R @ v for a 3x3 row-major matrix (np.matmul on a 3-vector). */
function matVec3(m: ArrayLike<number>, v: ArrayLike<number>): Float64Array {
  const out = new Float64Array(3)
  out[0] = m[0] * v[0] + m[1] * v[1] + m[2] * v[2]
  out[1] = m[3] * v[0] + m[4] * v[1] + m[5] * v[2]
  out[2] = m[6] * v[0] + m[7] * v[1] + m[8] * v[2]
  return out
}

/** np.cross for 3-vectors. */
function cross3(a: ArrayLike<number>, b: ArrayLike<number>): Float64Array {
  const out = new Float64Array(3)
  out[0] = a[1] * b[2] - a[2] * b[1]
  out[1] = a[2] * b[0] - a[0] * b[2]
  out[2] = a[0] * b[1] - a[1] * b[0]
  return out
}

/** v / np.linalg.norm(v). */
function normalize3(a: ArrayLike<number>): Float64Array {
  const n = Math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])
  const out = new Float64Array(3)
  out[0] = a[0] / n
  out[1] = a[1] / n
  out[2] = a[2] / n
  return out
}

/** Raw image pixels: uint8 RGB, row-major, `rows * cols * 3` entries. */
export interface ImageArray {
  readonly data: Uint8Array
  readonly rows: number
  readonly cols: number
}

/** A 2D point in pixel units. */
export interface PointPx {
  readonly x: number
  readonly y: number
}

/** Pinhole camera intrinsics. The browser always estimates these. */
export interface CameraIntrinsics {
  readonly focalLengthPx: number
  readonly principalPoint: PointPx
}

/**
 * Generic 3D face model for head pose:
 *
 * Taken from MediaPipe's canonical face model (google-ai-edge/mediapipe,
 * canonical_face_model.obj) reference points. Coordinates are [x,y,z]
 * where y is up, z faces the camera. Units are centimeters
 *
 * These values are what we would expect from a generic face
 * looking straight into the camera
 */
export const GENERIC_FACE_MODEL = new Float64Array([
  -4.446, -2.664, -3.173, // right eye, outer corner (MediaPipe landmark 33)
  -1.856, -2.585, -3.758, // right eye, inner corner (MediaPipe landmark 133)
  1.856, -2.585, -3.758, // left eye, inner corner  (MediaPipe landmark 362)
  4.446, -2.664, -3.173, // left eye, outer corner  (MediaPipe landmark 263)
  -2.456, 4.343, -4.284, // mouth, right corner     (MediaPipe landmark 61)
  2.456, 4.343, -4.284, // mouth, left corner      (MediaPipe landmark 291)
  0.0, 1.127, -7.476, // nose tip                (MediaPipe landmark 1)
  0.0, 9.403, -4.264, // chin                    (MediaPipe landmark 152)
])

// MediaPipe Face Mesh indices of key points we are interested in
export const MODEL_LANDMARK_IDS = [33, 133, 362, 263, 61, 291, 1, 152] as const

// The same model file `get_landmarker` fetches in Python.
const LANDMARKER_MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/face_landmarker/' +
  'face_landmarker/float16/latest/face_landmarker.task'

// The wasm runtime must come from the same @mediapipe/tasks-vision version
// this module is compiled against (MediaPipe pins loader and binary).
const TASKS_VISION_VERSION = '0.10.22-rc.20250304'

let cachedLandmarker: FaceLandmarker | null = null
let cachedLandmarkerPromise: Promise<FaceLandmarker> | null = null

/**
 * Create the FaceLandmarker.
 *
 * The Python function downloads `landmarker.task` on first use and keeps it
 * at model/models/landmarker.task; in the browser the file is fetched from
 * the same URL (or a caller-provided one) and the landmarker instance is
 * cached for the lifetime of the page.
 */
export async function getLandmarker(
  modelAssetPath: string = LANDMARKER_MODEL_URL,
): Promise<FaceLandmarker> {
  if (cachedLandmarker) return cachedLandmarker
  if (cachedLandmarkerPromise) return cachedLandmarkerPromise

  cachedLandmarkerPromise = (async () => {
    const vision = await import('@mediapipe/tasks-vision')
    const fileset = await vision.FilesetResolver.forVisionTasks(
      `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${TASKS_VISION_VERSION}/wasm`,
    )
    const landmarker = await vision.FaceLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath, delegate: 'CPU' },
      runningMode: 'IMAGE',
      numFaces: 1,
      minFaceDetectionConfidence: 0.5,
    })
    cachedLandmarker = landmarker
    return landmarker
  })()

  return cachedLandmarkerPromise
}

/** Detect facial landmarks in an RGB uint8 image. */
export function landmarkDetection(
  landmarker: FaceLandmarker,
  image: ImageArray,
): PointPx[] | null {
  const rgba = new Uint8ClampedArray(image.rows * image.cols * 4)
  for (let i = 0, j = 0; i < image.rows * image.cols * 3; i += 3, j += 4) {
    rgba[j] = image.data[i]
    rgba[j + 1] = image.data[i + 1]
    rgba[j + 2] = image.data[i + 2]
    rgba[j + 3] = 255
  }
  const mpImage = new ImageData(rgba, image.cols, image.rows)
  const result = landmarker.detect(mpImage)
  if (result.faceLandmarks.length === 0) return null
  const [face] = result.faceLandmarks // num_faces=1 -> exactly zero or one
  const height = image.rows
  const width = image.cols
  return face.map((landmark) => ({ x: landmark.x * width, y: landmark.y * height }))
}

/**
 * Estimate head pose by computing a rotation + translation matrix from a
 * generic 3D face to the observed subject face.
 *
 * Returns [R, t] mapping the 3D model into the OpenCV camera frame
 * (+X image right, +Y image down, +Z forward): R is the 3x3 rotation
 * matrix, t the model origin's translation in the model's units
 * (~centimetres; expect t[2] ~ 50-60 for MPIIFaceGaze webcam distance).
 * A face looking straight at the camera gives R ~ identity.
 */
export async function headPoseEst(
  landmarks: PointPx[],
  intrinsics: CameraIntrinsics,
): Promise<[Float64Array, Float64Array]> {
  // We need to ensure our landmark feature vector has all of the features we want
  const maxId = Math.max(...MODEL_LANDMARK_IDS)
  if (landmarks.length < maxId + 1) {
    throw new Error(
      `head pose needs the full Face Mesh (>= ${maxId + 1} points), got ${landmarks.length}`,
    )
  }

  // Extract only the points we care about
  const imagePoints = new Float64Array(MODEL_LANDMARK_IDS.length * 2)
  MODEL_LANDMARK_IDS.forEach((id, i) => {
    imagePoints[2 * i] = landmarks[id].x
    imagePoints[2 * i + 1] = landmarks[id].y
  })

  // This func finds the 6 DOF rotation + translation vector (rvec, tvec)
  // between the subject's current pose and the "looking straight-on" pose
  const { rvec, tvec } = await solvePnP(
    GENERIC_FACE_MODEL,
    imagePoints,
    cameraMatrix(intrinsics),
  )

  // Vector -> matrix
  const rotation = await rodrigues(rvec)
  return [rotation, tvec]
}

function cameraMatrix(intrinsics: CameraIntrinsics): Float64Array {
  return new Float64Array([
    intrinsics.focalLengthPx, 0.0, intrinsics.principalPoint.x,
    0.0, intrinsics.focalLengthPx, intrinsics.principalPoint.y,
    0.0, 0.0, 1.0,
  ])
}

export interface NormalizedFace {
  /** 224x224 RGB face crop. */
  face: ImageArray
  /** 36x60 RGB crop of the subject's anatomical left eye. */
  leftEye: ImageArray
  /** 36x60 RGB crop of the subject's anatomical right eye. */
  rightEye: ImageArray
  /** The virtual-camera rotation R_norm (3x3 row-major). */
  RVirtual: Float64Array
}

/**
 * Warp the image into a canonical, head-centered frame.
 *
 * We place a virtual camera at a fixed position and normalize the face to
 * be at a fixed position away from the camera with a fixed rotation
 */
export async function normalizeFace(
  image: ImageArray,
  landmarks: PointPx[],
  R: Float64Array,
  t: Float64Array,
  intrinsics: CameraIntrinsics,
): Promise<NormalizedFace> {
  // Iris centres from the Face Mesh.
  const left_iris_id = 473
  const right_iris_id = 468

  // Check we have enough landmarks
  const maxId = Math.max(left_iris_id, right_iris_id)
  if (landmarks.length < maxId + 1) {
    throw new Error(
      `normalization needs the full refined Face Mesh (>= ${maxId + 1} points), got ${landmarks.length}`,
    )
  }

  // Fixed camera parameters for normalization. These
  // need be identical for every sample from every dataset
  // in both python and typescript
  const norm_dist = 60.0 // (cm)
  const norm_focal = 960.0 // (cm)
  const face_size = 224 // (px)

  // Virtual camera rotation:
  // z-axis = camera to face
  // x-axis = straight up
  const z_axis = normalize3(matVec3(R, [0.0, 0.0, 1.0]))
  const head_x = matVec3(R, [1.0, 0.0, 0.0])
  // this trick ensures that the x_axis is orthogonal with the z-axis,
  // even if it is not *exactly* the x rotation of the head
  const z_dot = head_x[0] * z_axis[0] + head_x[1] * z_axis[1] + head_x[2] * z_axis[2]
  const x_axis = normalize3([
    head_x[0] - z_dot * z_axis[0],
    head_x[1] - z_dot * z_axis[1],
    head_x[2] - z_dot * z_axis[2],
  ])
  const y_axis = cross3(z_axis, x_axis)
  // R_virtual = np.column_stack((x_axis, y_axis, z_axis))
  const R_virtual = new Float64Array(9)
  R_virtual[0] = x_axis[0]
  R_virtual[1] = y_axis[0]
  R_virtual[2] = z_axis[0]
  R_virtual[3] = x_axis[1]
  R_virtual[4] = y_axis[1]
  R_virtual[5] = z_axis[1]
  R_virtual[6] = x_axis[2]
  R_virtual[7] = y_axis[2]
  R_virtual[8] = z_axis[2]

  // Virtual camera center
  const C_virtual = new Float64Array(3)
  for (let i = 0; i < 3; i++) C_virtual[i] = t[i] - norm_dist * z_axis[i]

  const K = cameraMatrix(intrinsics)
  const K_norm = new Float64Array([
    norm_focal, 0.0, face_size / 2,
    0.0, norm_focal, face_size / 2,
    0.0, 0.0, 1.0,
  ])

  // H_norm is a translation matrix which answers: for each pixel captured by
  // a real camera, where would that be on on an image taken by our virtual camera
  // H_norm = norm_dist * K @ R_virtual @ np.linalg.inv(K_norm)
  //          + (K @ C_virtual)[:, None] * np.array([[0.0, 0.0, 1.0]])
  // (* and @ are left-associative in numpy: the scalar scaling comes first)
  const scaledK = new Float64Array(9)
  for (let i = 0; i < 9; i++) scaledK[i] = norm_dist * K[i]
  const scaledK_R = await matMul3x3(scaledK, R_virtual)
  const firstTerm = await matMul3x3(scaledK_R, await invert3x3(K_norm))
  const KC = matVec3(K, C_virtual)
  const H_norm = new Float64Array(9)
  for (let i = 0; i < 9; i++) {
    H_norm[i] = firstTerm[i]
  }
  // (K @ C_virtual)[:, None] * [[0, 0, 1]] adds column 2 scaled by KC
  H_norm[2] += KC[0]
  H_norm[5] += KC[1]
  H_norm[8] += KC[2]

  // Finally we can apply the rotations / translations to our assets
  const face = await warpPerspective(image, H_norm, { width: face_size, height: face_size })

  const H_src_to_dst = await invert3x3(H_norm)

  const eye_size = { height: 36, width: 60 } // (height px, width px)

  const warped = (point: PointPx): [number, number] => {
    const out = matVec3(H_src_to_dst, [point.x, point.y, 1.0])
    return [out[0] / out[2], out[1] / out[2]]
  }

  const left = await getRectSubPix(
    face,
    { width: eye_size.width, height: eye_size.height },
    { x: warped(landmarks[left_iris_id])[0], y: warped(landmarks[left_iris_id])[1] },
  )
  const right = await getRectSubPix(
    face,
    { width: eye_size.width, height: eye_size.height },
    { x: warped(landmarks[right_iris_id])[0], y: warped(landmarks[right_iris_id])[1] },
  )
  return { face, leftEye: left, rightEye: right, RVirtual: R_virtual }
}

/**
 * Convert a raw gaze label to match the normalized image.
 *
 * We output two formats of labels
 *   - a fraction of the screen (x/W, y/H)
 *   - Normalized gaze direction (the angle off of the z-axis)
 *
 * Inference has no ground truth; this exists for the calibration screen,
 * which must record targets in the same label space the model predicts in.
 */
export function transformLabel(
  gazeTargetPx: PointPx,
  screenSizePx: Size2i,
  RVirtual: ArrayLike<number>,
  gazeDirectionCam?: ArrayLike<number> | null,
): [fraction: { x: number; y: number }, direction: Float64Array | null] {
  const fraction = {
    x: gazeTargetPx.x / screenSizePx.width,
    y: gazeTargetPx.y / screenSizePx.height,
  }

  let direction: Float64Array | null = null
  if (gazeDirectionCam !== undefined && gazeDirectionCam !== null) {
    // direction = R_virtual.T @ gaze_direction_cam
    const d = new Float64Array(3)
    for (let i = 0; i < 3; i++) {
      d[i] =
        RVirtual[i] * gazeDirectionCam[0] +
        RVirtual[3 + i] * gazeDirectionCam[1] +
        RVirtual[6 + i] * gazeDirectionCam[2]
    }
    const length = Math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
    if (!Number.isFinite(length) || length === 0.0) {
      throw new Error('gaze direction must be a finite, non-zero vector')
    }
    d[0] /= length
    d[1] /= length
    d[2] /= length
    direction = d
  }

  return [fraction, direction]
}

/** The inference artifacts `preprocess` produces for one frame. */
export interface PreprocessedFrame {
  /** 224x224 normalized face crop (RGB uint8). */
  face: ImageArray
  /** 36x60 crop of the subject's anatomical left eye (RGB uint8). */
  leftEye: ImageArray
  /** 36x60 crop of the subject's anatomical right eye (RGB uint8). */
  rightEye: ImageArray
  /** The virtual-camera rotation R_norm (3x3 row-major). */
  RNorm: Float64Array
}

/** The reasons a frame's artifacts cannot be produced. */
export type DropReason =
  | 'face_not_detected'
  | 'pose_failed'
  | 'reprojection_error'
  | 'eyes_off_face'

/**
 * Run every preprocessing step on one frame.
 *
 * Quality checks first (cheapest first):
 * - face detection succeeds
 * - head pose solvable
 * - reprojection error
 * - iris points near their eye corners
 *
 * Frames passing all checks continue through face normalization.
 */
export async function preprocess(
  landmarker: FaceLandmarker,
  image: ImageArray,
  intrinsics: CameraIntrinsics,
  maxReproj = 0.1,
): Promise<PreprocessedFrame | DropReason> {
  const landmarks = landmarkDetection(landmarker, image)
  if (landmarks === null) return 'face_not_detected'

  let R: Float64Array
  let t: Float64Array
  try {
    ;[R, t] = await headPoseEst(landmarks, intrinsics)
  } catch {
    return 'pose_failed'
  }

  // reprojection error, normalized by the projected eye-corner distance
  const cameraFocal = intrinsics.focalLengthPx
  const cameraX = intrinsics.principalPoint.x
  const cameraY = intrinsics.principalPoint.y
  const K = new Float64Array([
    cameraFocal, 0.0, cameraX,
    0.0, cameraFocal, cameraY,
    0.0, 0.0, 1.0,
  ])
  const observed = new Float64Array(MODEL_LANDMARK_IDS.length * 2)
  MODEL_LANDMARK_IDS.forEach((id, i) => {
    observed[2 * i] = landmarks[id].x
    observed[2 * i + 1] = landmarks[id].y
  })
  const rvec = await rodriguesToVector(R)
  const projected = await projectPoints(GENERIC_FACE_MODEL, rvec, t, K)
  const eyeSpanDx = projected[0] - projected[2 * 3]
  const eyeSpanDy = projected[1] - projected[2 * 3 + 1]
  const eyeSpan = Math.sqrt(eyeSpanDx * eyeSpanDx + eyeSpanDy * eyeSpanDy)
  let reprojSum = 0
  for (let i = 0; i < MODEL_LANDMARK_IDS.length; i++) {
    const dx = projected[2 * i] - observed[2 * i]
    const dy = projected[2 * i + 1] - observed[2 * i + 1]
    reprojSum += Math.sqrt(dx * dx + dy * dy)
  }
  const reproj = reprojSum / MODEL_LANDMARK_IDS.length / Math.max(eyeSpan, 1e-9)
  if (reproj > maxReproj) return 'reprojection_error'

  // iris centres must sit near their eye corners' bounding box
  const irisChecks: Array<[number, number, number]> = [
    [468, 33, 133],
    [473, 263, 362],
  ]
  for (const [irisId, cornerA, cornerB] of irisChecks) {
    const a = landmarks[cornerA]
    const b = landmarks[cornerB]
    const iris = landmarks[irisId]
    const margin = 0.25 * Math.max(Math.abs(b.x - a.x), Math.abs(b.y - a.y))
    const inBox =
      Math.min(a.x, b.x) - margin <= iris.x &&
      iris.x <= Math.max(a.x, b.x) + margin &&
      Math.min(a.y, b.y) - margin <= iris.y &&
      iris.y <= Math.max(a.y, b.y) + margin
    if (!inBox) return 'eyes_off_face'
  }

  const { face, leftEye, rightEye, RVirtual } = await normalizeFace(
    image,
    landmarks,
    R,
    t,
    intrinsics,
  )
  return { face, leftEye, rightEye, RNorm: RVirtual }
}

/**
 * uint8 (B, H, W, 3) crops -> normalized float (B, 3, H, W) tensor data.
 *
 * Datasets hand back HWC uint8 because that is what OpenCV produces. The
 * conversion mirrors `prepare_images` from model/src/data/utils.py.
 *
 * Args:
 *   batch: stacked crops as uint8 RGB, HWC.
 *   imagenet: normalize with ImageNet statistics, which a model initialized
 *     from ImageNet weights expects. False scales to [0, 1] only, for a
 *     randomly initialized model.
 *
 * Returns a Float32Array in NCHW layout, ready for onnxruntime-web.
 */
export function prepareImages(
  batch: ImageArray[],
  { imagenet = true }: { imagenet?: boolean } = {},
): { data: Float32Array; shape: [number, number, number, number] } {
  const B = batch.length
  if (B === 0) throw new Error('batch must be non-empty')
  const H = batch[0].rows
  const W = batch[0].cols
  for (const img of batch) {
    if (img.rows !== H || img.cols !== W || img.data.length !== H * W * 3) {
      throw new Error(`expected (B, H, W, 3) with identical H, W`)
    }
  }
  const out = new Float32Array(B * 3 * H * W)
  const mean = [0.485, 0.456, 0.406]
  const std = [0.229, 0.224, 0.225]
  for (let b = 0; b < B; b++) {
    const img = batch[b]
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        for (let c = 0; c < 3; c++) {
          const v = img.data[(y * W + x) * 3 + c] / 255.0
          out[((b * 3 + c) * H + y) * W + x] = imagenet ? (v - mean[c]) / std[c] : v
        }
      }
    }
  }
  return { data: out, shape: [B, 3, H, W] }
}
