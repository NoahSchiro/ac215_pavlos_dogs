import { beforeEach, expect, test, vi } from 'vitest'
import { decodeCalibrationJpeg } from './jpeg'
import { getLandmarker } from '../vision/utils'
import { loadGazeModel, predictGaze } from '../vision/model'
import { predictCalibrationImage } from './inference'

// Browser decoding and network/WASM initialization are external boundaries.
vi.mock('./jpeg', () => ({ decodeCalibrationJpeg: vi.fn() }))
vi.mock('../vision/utils', () => ({ getLandmarker: vi.fn() }))
vi.mock('../vision/model', () => ({ loadGazeModel: vi.fn(), predictGaze: vi.fn() }))

const image = { rows: 480, cols: 640, data: new Uint8Array(640 * 480 * 3) }
const landmarker = {} as Awaited<ReturnType<typeof getLandmarker>>
const model = {} as Awaited<ReturnType<typeof loadGazeModel>>
const jpeg = new Blob(['jpeg'], { type: 'image/jpeg' })

beforeEach(() => {
  vi.resetAllMocks()
  vi.mocked(decodeCalibrationJpeg).mockResolvedValue(image)
  vi.mocked(getLandmarker).mockResolvedValue(landmarker)
  vi.mocked(loadGazeModel).mockResolvedValue(model)
  vi.mocked(predictGaze).mockResolvedValue({ x: 0.25, y: 0.75 })
})

test('predicts from the decoded image using image dimensions for camera parameters', async () => {
  expect(await predictCalibrationImage(jpeg)).toEqual({
    ok: true,
    prediction: { x: 0.25, y: 0.75 },
    image: { width: 640, height: 480 },
    intrinsics: { focalLengthPx: 640, principalPoint: { x: 320, y: 240 } },
  })
  expect(decodeCalibrationJpeg).toHaveBeenCalledWith(jpeg)
  expect(predictGaze).toHaveBeenCalledWith(landmarker, model, image, {
    focalLengthPx: 640, principalPoint: { x: 320, y: 240 },
  })
})

test.each(['face_not_detected', 'pose_failed', 'reprojection_error', 'eyes_off_face'] as const)(
  'preserves the preprocessing rejection %s', async (reason) => {
    vi.mocked(predictGaze).mockResolvedValue(reason)
    expect(await predictCalibrationImage(jpeg)).toEqual({ ok: false, stage: 'preprocess', reason })
  },
)

test.each([
  ['decode', decodeCalibrationJpeg],
  ['landmarker', getLandmarker],
  ['model', loadGazeModel],
  ['inference', predictGaze],
] as const)('identifies a %s failure and preserves its message', async (stage, operation) => {
  vi.mocked(operation).mockRejectedValue(new Error('resource unavailable'))
  expect(await predictCalibrationImage(jpeg)).toEqual({
    ok: false, stage, reason: 'resource unavailable',
  })
  if (stage !== 'inference') expect(predictGaze).not.toHaveBeenCalled()
})

test.each([{ x: NaN, y: 0.5 }, { x: 0.5, y: Infinity }])(
  'rejects non-finite predictions: %j', async (point) => {
    vi.mocked(predictGaze).mockResolvedValue(point)
    expect(await predictCalibrationImage(jpeg)).toEqual({
      ok: false, stage: 'inference', reason: 'non_finite_prediction',
    })
  },
)
