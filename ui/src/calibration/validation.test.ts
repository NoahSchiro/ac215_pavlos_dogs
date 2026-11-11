import { beforeEach, expect, test, vi } from 'vitest'
import { predictCalibrationImage, type CalibrationImagePrediction } from './inference'
import { TEST_CAMERA, testSession } from './fixtures/testSession'
import { VALIDATION_POINTS, type CalibrationResult } from './points'
import { validateCalibrationCapture } from './validation'

vi.mock('./inference', () => ({ predictCalibrationImage: vi.fn() }))
const predict = vi.mocked(predictCalibrationImage)
const result = (x: number, y: number): CalibrationImagePrediction => ({
  ok: true, prediction: { x, y }, image: { width: 640, height: 480 },
  intrinsics: { focalLengthPx: 640, principalPoint: { x: 320, y: 240 } },
})
function capture(): CalibrationResult {
  return { completedAt: 20, camera: TEST_CAMERA, samples: VALIDATION_POINTS.flatMap((target, pointIndex) =>
    Array.from({ length: 3 }, () => ({ image: new Blob(['fresh']), target, pointIndex,
      timestamp: 10, viewportWidth: 800, viewportHeight: 600 }))) }
}
beforeEach(() => predict.mockReset())

test('compares the same fresh frames with the fixed fit and leaves the saved session unchanged', async () => {
  const session = testSession()
  const before = JSON.stringify(session)
  const images = capture()
  images.samples.forEach(({ target }) => predict.mockResolvedValueOnce(result(1 - target.x, target.y)))
  const summary = await validateCalibrationCapture(images, session)
  expect(summary.validFrameCount).toBe(12)
  expect(summary.rawMeanError).toBeCloseTo(0.5, 12)
  expect(summary.correctedMeanError).toBeCloseTo(0, 12)
  expect(JSON.stringify(session)).toBe(before)
})

test('rejects validation in a different viewport before inference', async () => {
  const images = capture()
  predict.mockResolvedValue(result(0.5, 0.5))
  images.samples.forEach((sample) => Object.assign(sample, { viewportWidth: 1200 }))
  await expect(validateCalibrationCapture(images, testSession())).rejects.toThrow(/window|viewport/i)
  expect(predict).not.toHaveBeenCalled()
})

test('excludes a rejected frame from both means and requires every target', async () => {
  const images = capture()
  images.samples.forEach(({ target }, i) => predict.mockResolvedValueOnce(i === 0
    ? { ok: false, stage: 'preprocess', reason: 'face_not_detected' } : result(1 - target.x, target.y)))
  const summary = await validateCalibrationCapture(images, testSession())
  expect(summary.validFrameCount).toBe(11)
  expect(summary.rawMeanError).toBeCloseTo(0.5, 12)
  expect(summary.correctedMeanError).toBeCloseTo(0, 12)
  images.samples.forEach(({ target }, i) => predict.mockResolvedValueOnce(i < 3
    ? { ok: false, stage: 'preprocess', reason: 'pose_failed' } : result(target.x, target.y)))
  await expect(validateCalibrationCapture(images, testSession())).rejects.toThrow(/target.*1/i)
})

test('reports worse correction without clipping the accuracy metric', async () => {
  const session = testSession()
  const fit = { weight: [[1, 0], [0, 1]] as const, intercept: [2, 0] as const, lambda: 0.1 }
  const images = capture()
  images.samples.forEach(({ target }) => predict.mockResolvedValueOnce(result(target.x, target.y)))
  const summary = await validateCalibrationCapture(images, { ...session, record: { ...session.record,
    calibration: { ...session.record.calibration, fit } } })
  expect(summary.rawMeanError).toBe(0)
  expect(summary.correctedMeanError).toBeCloseTo(2, 12)
})

test.each(['pending', 'camera', 'old', 'target'] as const)('rejects invalid validation input before inference: %s', async (kind) => {
  const session = testSession()
  const images = capture()
  const samples = [...images.samples]
  if (kind === 'old') samples[0] = { ...samples[0], timestamp: 1 }
  if (kind === 'target') samples[0] = { ...samples[0], target: { x: 0.1, y: 0.1 } }
  await expect(validateCalibrationCapture({ ...images, samples,
    camera: { ...TEST_CAMERA, deviceId: kind === 'camera' ? 'other' : TEST_CAMERA.deviceId },
  }, { ...session, cameraVerification: kind === 'pending' ? 'pending' : 'verified' })).rejects.toThrow(/camera|fresh|target/i)
  expect(predict).not.toHaveBeenCalled()
})

test('cancellation stops after an in-flight frame without publishing progress', async () => {
  let finish!: (value: CalibrationImagePrediction) => void
  predict.mockReturnValueOnce(new Promise((resolve) => { finish = resolve }))
  const controller = new AbortController()
  const progress = vi.fn()
  const pending = validateCalibrationCapture(capture(), testSession(), { signal: controller.signal, onProgress: progress })
  const rejected = expect(pending).rejects.toMatchObject({ name: 'AbortError' })
  controller.abort()
  finish(result(0.2, 0.3))
  await rejected
  expect(predict).toHaveBeenCalledTimes(1)
  expect(progress).not.toHaveBeenCalled()
})
