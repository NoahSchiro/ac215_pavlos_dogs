/**
 * Tests for browser GazeNet inference (onnxruntime-web).
 *
 * These verify the forward pass, deliberately independent of the trained
 * weights: the served ONNX must load with the GazeNet interface, consume the
 * preprocessing artifacts, and return sigmoid screen fractions, so a
 * retrained model swaps in without test changes. How accurate the
 * predictions are is measured in model/ during training, not here.
 */

import { expect, test } from 'vitest'
import faceUrl from './fixtures/face.jpg'
import faceCropUrl from './fixtures/face_crop.bin?url'
import leftCropUrl from './fixtures/left_crop.bin?url'
import rightCropUrl from './fixtures/right_crop.bin?url'
import { loadGazeModel, predictGaze, predictScreenFraction } from './model'
import { getLandmarker, type CameraIntrinsics, type ImageArray } from './utils'

const MPII_INTRINSICS: CameraIntrinsics = {
  focalLengthPx: 996.4509,
  principalPoint: { x: 624.6634, y: 364.0874 },
}

async function loadBin(url: string, rows: number, cols: number): Promise<ImageArray> {
  const data = new Uint8Array(await (await fetch(url)).arrayBuffer())
  expect(data.length).toBe(rows * cols * 3)
  return { data, rows, cols }
}

async function loadFaceImage(): Promise<ImageArray> {
  const blob = await (await fetch(faceUrl)).blob()
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

const modelPromise = loadGazeModel()
const cropsPromise = Promise.all([
  loadBin(faceCropUrl, 224, 224),
  loadBin(leftCropUrl, 36, 60),
  loadBin(rightCropUrl, 36, 60),
])

test('the served model has the GazeNet interface', async () => {
  const model = await modelPromise
  expect(model.inputNames).toEqual(['face', 'left_eye', 'right_eye'])
  expect(model.outputName).toBe('prediction')
})

test('predictScreenFraction runs a deterministic forward pass', async () => {
  const model = await modelPromise
  const [face, left, right] = await cropsPromise
  const frame = { face, leftEye: left, rightEye: right, RNorm: new Float64Array(9) }

  const first = await predictScreenFraction(model, frame)
  const second = await predictScreenFraction(model, frame)

  // sigmoid-activated screen fractions, finite and in range
  for (const p of [first, second]) {
    expect(p.x).toBeGreaterThanOrEqual(0)
    expect(p.x).toBeLessThanOrEqual(1)
    expect(p.y).toBeGreaterThanOrEqual(0)
    expect(p.y).toBeLessThanOrEqual(1)
  }
  // inference on the CPU wasm backend is deterministic for fixed inputs
  expect(first).toEqual(second)
})

test('predictGaze returns a screen fraction for a face frame', async () => {
  const model = await modelPromise
  const landmarker = await getLandmarker()
  const image = await loadFaceImage()

  const prediction = await predictGaze(landmarker, model, image, MPII_INTRINSICS)

  expect(prediction).not.toBe('face_not_detected')
  expect(typeof prediction).toBe('object')
  const { x, y } = prediction as { x: number; y: number }
  expect(x).toBeGreaterThanOrEqual(0)
  expect(x).toBeLessThanOrEqual(1)
  expect(y).toBeGreaterThanOrEqual(0)
  expect(y).toBeLessThanOrEqual(1)
})

test('predictGaze reports a drop reason for frames without a face', async () => {
  const model = await modelPromise
  const landmarker = await getLandmarker()
  // seeded xorshift noise, face-free by construction (mirrors utils.test.ts)
  const rows = 480
  const cols = 640
  const noise = new Uint8Array(rows * cols * 3)
  let state = 0x9e3779b9
  for (let i = 0; i < noise.length; i++) {
    state ^= state << 13
    state ^= state >>> 17
    state ^= state << 5
    noise[i] = (state >>> 24) & 0xff
  }

  const result = await predictGaze(
    landmarker,
    model,
    { data: noise, rows, cols },
    MPII_INTRINSICS,
  )

  expect(result).toBe('face_not_detected')
})
