import { beforeEach, expect, test, vi } from 'vitest'
import { predictCalibrationImage, type CalibrationImagePrediction } from './inference'
import { TEST_CAMERA, testSession } from './fixtures/testSession'
import { VALIDATION_POINTS, validateCalibrationCapture } from './validation'
import type { CalibrationResult } from './points'
import * as ridge from './ridge'

vi.mock('./inference', () => ({ predictCalibrationImage: vi.fn() }))
const predict = vi.mocked(predictCalibrationImage)

function captured(): CalibrationResult {
  return {
    completedAt: 20, camera: TEST_CAMERA,
    samples: VALIDATION_POINTS.flatMap((target, pointIndex) => Array.from({ length: 3 }, () => ({
      image: new Blob(['fresh-jpeg'], { type: 'image/jpeg' }), target, pointIndex,
      timestamp: 10, viewportWidth: 800, viewportHeight: 600,
    }))),
  }
}
const success = (x: number, y: number): CalibrationImagePrediction => ({
  ok: true, prediction: { x, y }, image: { width: 640, height: 480 },
  intrinsics: { focalLengthPx: 640, principalPoint: { x: 320, y: 240 } },
})
beforeEach(() => { vi.restoreAllMocks(); predict.mockReset() })

test('scores the same held-out frames before/after correction without refitting or changing the fit', async () => {
  const capture = captured()
  const session = testSession()
  capture.samples.forEach(({ target }) => predict.mockResolvedValueOnce(success(1 - target.x, target.y)))
  const fit = vi.spyOn(ridge, 'fitCalibration')
  const before = JSON.stringify(session)
  const onProgress = vi.fn()
  const report = await validateCalibrationCapture(capture, session, { onProgress })
  expect(report.validFrameCount).toBe(12)
  expect(report.validFrameCounts).toEqual([3, 3, 3, 3])
  expect(report.rawMeanError).toBeCloseTo(0.5, 12)
  expect(report.correctedMeanError).toBeCloseTo(0, 12)
  expect(report.improved).toBe(true)
  expect(report.metric).toBe('mean-euclidean-normalized-unclipped')
  expect(report.camera).toEqual(TEST_CAMERA)
  expect(report.compatibility).toEqual(session.record.compatibility)
  expect(report.fit).toEqual(session.record.calibration.fit)
  expect(fit).not.toHaveBeenCalled()
  expect(JSON.stringify(session)).toBe(before)
  expect(onProgress).toHaveBeenLastCalledWith({ completed: 12, total: 12 })
})

test('excludes a rejected frame from both means and records its failure', async () => {
  const capture = captured()
  capture.samples.forEach(({ target }, i) => predict.mockResolvedValueOnce(
    i === 0 ? { ok: false, stage: 'preprocess', reason: 'face_not_detected' } : success(1 - target.x, target.y),
  ))
  const report = await validateCalibrationCapture(capture, testSession())
  expect(report.validFrameCount).toBe(11)
  expect(report.validFrameCounts).toEqual([2, 3, 3, 3])
  expect(report.rawMeanError).toBeCloseTo(0.5, 12)
  expect(report.frames[0].failure).toContain('face_not_detected')
})

test('reports non-improvement and does not clip the primary accuracy metric', async () => {
  const capture = captured()
  const session = testSession()
  const fit = { weight: [[1, 0], [0, 1]] as const, intercept: [2, 0] as const, lambda: 0.1 }
  capture.samples.forEach(({ target }) => predict.mockResolvedValueOnce(success(target.x, target.y)))
  const report = await validateCalibrationCapture(capture, { ...session, record: { ...session.record, calibration: { ...session.record.calibration, fit } } })
  expect(report.rawMeanError).toBe(0)
  expect(report.correctedMeanError).toBeCloseTo(2, 12)
  expect(report.improved).toBe(false)
})

test('requires valid coverage at every fixed validation target', async () => {
  captured().samples.forEach(({ target }, i) => predict.mockResolvedValueOnce(
    i < 3 ? { ok: false, stage: 'preprocess', reason: 'pose_failed' } : success(target.x, target.y),
  ))
  await expect(validateCalibrationCapture(captured(), testSession())).rejects.toThrow(/target.*1/)
})

test.each(['pending', 'camera', 'training-blob', 'old-time', 'wrong-target', 'wrong-count', 'non-finite-time'] as const)(
  'rejects an invalid validation session/capture: %s', async (kind) => {
    const capture = captured()
    const session = testSession()
    const samples = [...capture.samples]
    if (kind === 'training-blob') samples[0] = { ...samples[0], image: session.record.calibration.capture.samples[0].image }
    if (kind === 'old-time') samples[0] = { ...samples[0], timestamp: 1 }
    if (kind === 'wrong-target') samples[0] = { ...samples[0], target: { x: 0.1, y: 0.1 } }
    if (kind === 'wrong-count') samples.pop()
    await expect(validateCalibrationCapture({ ...capture, samples,
      completedAt: kind === 'non-finite-time' ? NaN : capture.completedAt,
      camera: { ...TEST_CAMERA, deviceId: kind === 'camera' ? 'other' : TEST_CAMERA.deviceId },
    }, { ...session, cameraVerification: kind === 'pending' ? 'pending' : 'verified' })).rejects.toThrow(/Verify|fresh|fixed/)
    expect(predict).not.toHaveBeenCalled()
  },
)

test('cancellation stops evaluation after the current frame without publishing a report', async () => {
  let finish!: (value: CalibrationImagePrediction) => void
  predict.mockReturnValueOnce(new Promise((resolve) => { finish = resolve }))
  const controller = new AbortController()
  const onProgress = vi.fn()
  const pending = validateCalibrationCapture(captured(), testSession(), { signal: controller.signal, onProgress })
  const rejected = expect(pending).rejects.toMatchObject({ name: 'AbortError' })
  controller.abort()
  finish(success(0.75, 0.25))
  await rejected
  expect(predict).toHaveBeenCalledTimes(1)
  expect(onProgress).toHaveBeenCalledTimes(1)
})
