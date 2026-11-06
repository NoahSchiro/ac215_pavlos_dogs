/**
 * Tests for landmark detection and the geometric pipeline.
 *
 * Line-for-line port of `model/tests/test_utils.py`. The only structural
 * deviations are forced by the platform:
 * - images load through the browser image pipeline instead of PIL, and are
 *   indexed with a `pixel()` helper instead of numpy subscripts
 * - the face landmarker is async (wasm + model fetch)
 * - the noise image for `test_returns_none_without_face` uses a seeded
 *   xorshift instead of numpy's PCG64; any face-free noise works
 */

import { expect, test } from 'vitest'
import faceUrl from './fixtures/face.jpg'
import {
  GENERIC_FACE_MODEL,
  MODEL_LANDMARK_IDS,
  getLandmarker,
  headPoseEst,
  landmarkDetection,
  normalizeFace,
  transformLabel,
  type CameraIntrinsics,
  type ImageArray,
  type PointPx,
} from './utils'
import { projectPoints, rodrigues } from './cv'

// I pulled this image from MPII
// FACE_IMAGE_PATH = Path(__file__).parent / "data" / "face.jpg"
const FACE_IMAGE_PATH = faceUrl

async function loadFaceImage(): Promise<ImageArray> {
  const blob = await (await fetch(FACE_IMAGE_PATH)).blob()
  const bitmap = await createImageBitmap(blob)
  const canvas = new OffscreenCanvas(bitmap.width, bitmap.height)
  const ctx = canvas.getContext('2d') as OffscreenCanvasRenderingContext2D
  ctx.drawImage(bitmap, 0, 0)
  const rgba = ctx.getImageData(0, 0, bitmap.width, bitmap.height).data
  const rgb = new Uint8Array(bitmap.width * bitmap.height * 3)
  for (let i = 0, j = 0; i < rgb.length; i += 3, j += 4) {
    rgb[i] = rgba[j]
    rgb[i + 1] = rgba[j + 1]
    rgb[i + 2] = rgba[j + 2]
  }
  return { data: rgb, rows: bitmap.height, cols: bitmap.width }
}

/** np.allclose with the defaults the Python tests rely on (atol given). */
function expectAllClose(
  actual: ArrayLike<number>,
  expected: ArrayLike<number>,
  atol: number,
) {
  expect(actual.length).toBe(expected.length)
  for (let i = 0; i < actual.length; i++) {
    expect(Math.abs(actual[i] - expected[i])).toBeLessThanOrEqual(atol)
  }
}

const pixel = (
  image: ImageArray,
  y: number,
  x: number,
): [number, number, number] => {
  const ofs = (y * image.cols + x) * 3
  return [image.data[ofs], image.data[ofs + 1], image.data[ofs + 2]]
}

/** ndarray.max() equivalent (spread breaks the stack on large arrays). */
function maxOf(data: Uint8Array): number {
  let max = 0
  for (let i = 0; i < data.length; i++) if (data[i] > max) max = data[i]
  return max
}

test('detects landmarks on face', async () => {
  const image = await loadFaceImage()
  const model = await getLandmarker()
  const landmarks = landmarkDetection(model, image)

  expect(landmarks).not.toBeNull()
  // 468 face mesh points + 10 iris points
  expect(landmarks).toHaveLength(478)
  expect(landmarks?.every((point) => typeof point.x === 'number' && typeof point.y === 'number')).toBe(true)
})

test('landmarks are in pixel coordinates', async () => {
  const image = await loadFaceImage()
  const model = await getLandmarker()
  const landmarks = landmarkDetection(model, image)

  expect(landmarks).not.toBeNull()
  const height = image.rows
  const width = image.cols
  expect(landmarks?.every((point) => 0 <= point.x && point.x <= width)).toBe(true)
  expect(landmarks?.every((point) => 0 <= point.y && point.y <= height)).toBe(true)
})

test('eye landmark anatomy', async () => {
  // Subject's right eye (33) must sit image-left of their left eye (263).
  const image = await loadFaceImage()
  const model = await getLandmarker()
  const landmarks = landmarkDetection(model, image)

  expect(landmarks).not.toBeNull()
  const height = image.rows
  expect(landmarks![33].x).toBeLessThan(landmarks![263].x)
  expect(Math.abs(landmarks![33].y - landmarks![263].y)).toBeLessThanOrEqual(height * 0.05)
})

test('detection is deterministic', async () => {
  const image = await loadFaceImage()
  const model = await getLandmarker()

  const first = landmarkDetection(model, image)
  const second = landmarkDetection(model, image)

  expect(first).not.toBeNull()
  expect(second).not.toBeNull()
  expect(second).toEqual(first)
})

test('returns none without face', async () => {
  // noise = np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)
  const rows = 480
  const cols = 640
  const noise = new Uint8Array(rows * cols * 3)
  let state = 0x9e3779b9 // seeded xorshift; the noise only has to be face-free
  for (let i = 0; i < noise.length; i++) {
    state ^= state << 13
    state ^= state >>> 17
    state ^= state << 5
    noise[i] = (state >>> 24) & 0xff
  }
  const model = await getLandmarker()

  expect(landmarkDetection(model, { data: noise, rows, cols })).toBeNull()
})

async function makeLandmarksFromProjection(
  rvec: ArrayLike<number>,
  tvec: ArrayLike<number>,
  cameraMatrix: ArrayLike<number>,
): Promise<PointPx[]> {
  // Apply rotation / translation vectors to generic 3D face.
  const projected = await projectPoints(GENERIC_FACE_MODEL, rvec, tvec, cameraMatrix)
  const landmarks: PointPx[] = Array.from({ length: 478 }, () => ({ x: 0.0, y: 0.0 }))
  MODEL_LANDMARK_IDS.forEach((id, i) => {
    landmarks[id] = { x: projected[2 * i], y: projected[2 * i + 1] }
  })
  return landmarks
}

test('recovers known pose from synthetic projection', async () => {
  // solvePnP must invert the projection of a known pose exactly.
  const intrinsics: CameraIntrinsics = {
    focalLengthPx: 1000.0,
    principalPoint: { x: 640.0, y: 360.0 },
  }
  const cameraMatrix = [1000.0, 0.0, 640.0, 0.0, 1000.0, 360.0, 0.0, 0.0, 1.0]
  const rvecTrue = [0.15, -0.25, 0.05] // yaw/pitch/roll
  const tvecTrue = [3.0, -4.0, 60.0] // x/y/z

  const landmarks = await makeLandmarksFromProjection(rvecTrue, tvecTrue, cameraMatrix)
  const [rMat, tMat] = await headPoseEst(landmarks, intrinsics)

  const RTrue = await rodrigues(rvecTrue)
  expectAllClose(rMat, RTrue, 1e-6)
  expectAllClose(tMat, tvecTrue, 1e-6)
})

test('fronto parallel face gives identity rotation', async () => {
  const intrinsics: CameraIntrinsics = {
    focalLengthPx: 1000.0,
    principalPoint: { x: 0.0, y: 0.0 },
  }
  // Face with no rotation, 60 cm in front of camera
  const landmarks = await makeLandmarksFromProjection(
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 60.0],
    [1000.0, 0.0, 0.0, 0.0, 1000.0, 0.0, 0.0, 0.0, 1000.0],
  )

  const [rMat, tMat] = await headPoseEst(landmarks, intrinsics)

  expectAllClose(rMat, [1, 0, 0, 0, 1, 0, 0, 0, 1], 1e-6)
  expectAllClose(tMat, [0.0, 0.0, 60.0], 1e-6)
})

test('rejects landmark lists without the model points', async () => {
  const intrinsics: CameraIntrinsics = {
    focalLengthPx: 1000.0,
    principalPoint: { x: 0.0, y: 0.0 },
  }

  await expect(
    headPoseEst(
      Array.from({ length: 10 }, () => ({ x: 0.0, y: 0.0 })),
      intrinsics,
    ),
  ).rejects.toThrow(/Face Mesh/)
})

const NORM_INTRINSICS: CameraIntrinsics = {
  focalLengthPx: 1000.0,
  principalPoint: { x: 640.0, y: 360.0 },
}
const NORM_K = [1000.0, 0.0, 640.0, 0.0, 1000.0, 360.0, 0.0, 0.0, 1.0]

function blackCanvas(): ImageArray {
  return { data: new Uint8Array(720 * 1280 * 3), rows: 720, cols: 1280 }
}

function drawRedSquare(image: ImageArray, x: number, y: number, half = 4): void {
  // image[int(y) - half : int(y) + half, int(x) - half : int(x) + half] = (255, 0, 0)
  for (let yy = Math.trunc(y) - half; yy < Math.trunc(y) + half; yy++) {
    for (let xx = Math.trunc(x) - half; xx < Math.trunc(x) + half; xx++) {
      const ofs = (yy * image.cols + xx) * 3
      image.data[ofs] = 255
      image.data[ofs + 1] = 0
      image.data[ofs + 2] = 0
    }
  }
}

test('eye crops follow anatomical convention', async () => {
  // Subject's LEFT eye (image-right in non-mirrored frames) -> `left_eye`.
  const t = [0.0, 0.0, 60.0]
  const landmarks = await makeLandmarksFromProjection([0.0, 0.0, 0.0], t, NORM_K)
  // iris positions ~ +-2.88 units off-centre, projected by K at depth 60
  landmarks[473] = { x: 688.0, y: 316.4 } // subject's left -> image right
  landmarks[468] = { x: 592.0, y: 316.4 } // subject's right -> image left

  const image = blackCanvas()
  drawRedSquare(image, 688.0, 316.4) // marker on the subject's left eye only

  const { leftEye, rightEye } = await normalizeFace(
    image,
    landmarks,
    new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]),
    new Float64Array(t),
    NORM_INTRINSICS,
  )

  // crops centre on the warped iris positions (158, 70) and (66, 70)
  expect(pixel(leftEye, 18, 30)).toEqual([255, 0, 0])
  expect(pixel(rightEye, 18, 30)).toEqual([0, 0, 0])
})

test('real face normalization produces valid crops', async () => {
  const image = await loadFaceImage()
  const model = await getLandmarker()
  const landmarks = landmarkDetection(model, image)

  expect(landmarks).not.toBeNull()
  const intrinsics: CameraIntrinsics = {
    focalLengthPx: 996.4509,
    principalPoint: { x: 624.6634, y: 364.0874 },
  }
  const [R, t] = await headPoseEst(landmarks!, intrinsics)

  const { face, leftEye, rightEye } = await normalizeFace(image, landmarks!, R, t, intrinsics)

  expect(face.rows).toBe(224)
  expect(face.cols).toBe(224)
  expect(leftEye.rows).toBe(36)
  expect(leftEye.cols).toBe(60)
  expect(rightEye.rows).toBe(36)
  expect(rightEye.cols).toBe(60)
  expect(maxOf(face.data)).toBeGreaterThan(0)
  expect(maxOf(leftEye.data)).toBeGreaterThan(0)
  expect(maxOf(rightEye.data)).toBeGreaterThan(0)
  expect(leftEye.data).not.toEqual(rightEye.data)

  const second = await normalizeFace(image, landmarks!, R, t, intrinsics)
  expect(second.face.data).toEqual(face.data)
  expect(second.leftEye.data).toEqual(leftEye.data)
  expect(second.rightEye.data).toEqual(rightEye.data)
})

test('screen fraction is resolution independent', () => {
  const [fraction, direction] = transformLabel(
    { x: 562.5, y: 1218.0 },
    { width: 1125, height: 2436 },
    new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]),
  )

  expect(fraction.x).toBeCloseTo(0.5)
  expect(fraction.y).toBeCloseTo(0.5)
  expect(direction).toBeNull()
})

test('out of bounds targets pass through unclamped', () => {
  const [fraction] = transformLabel(
    { x: -10.0, y: 5000.0 },
    { width: 1125, height: 2436 },
    new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]),
  )

  expect(fraction.x).toBeCloseTo(-10.0 / 1125.0)
  expect(fraction.y).toBeCloseTo(5000.0 / 2436.0)
})

test('direction is rotated into normalized frame', () => {
  // R_virtual is a +90-degree rotation about z. Its transpose (what the
  // function applies) rotates the camera x-axis into the normalized -y-axis
  const RVirtual = [0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]

  const [fraction, direction] = transformLabel(
    { x: 0.0, y: 0.0 },
    { width: 100, height: 100 },
    RVirtual,
    [2.0, 0.0, 0.0], // unnormalized on purpose
  )

  expect(fraction).toEqual({ x: 0.0, y: 0.0 })
  expect(direction).not.toBeNull()
  expectAllClose(direction!, [0.0, -1.0, 0.0], 1e-12)
  const norm = Math.sqrt(
    direction![0] ** 2 + direction![1] ** 2 + direction![2] ** 2,
  )
  expect(Math.abs(norm - 1.0)).toBeLessThanOrEqual(1e-12)
})

test('rejects degenerate direction', () => {
  expect(() =>
    transformLabel(
      { x: 0.0, y: 0.0 },
      { width: 100, height: 100 },
      new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]),
      [0, 0, 0],
    ),
  ).toThrow(/non-zero/)
})
