import { beforeEach, expect, test, vi } from 'vitest'
import { predictCalibrationImage, type CalibrationImagePrediction } from './inference'
import { CALIBRATION_POINTS, type CalibrationResult } from './points'
import { processCalibration } from './process'

vi.mock('./inference', () => ({ predictCalibrationImage: vi.fn() }))
const predict = vi.mocked(predictCalibrationImage)

function capture(perPoint = 3): CalibrationResult {
  return {
    completedAt: 123,
    samples: CALIBRATION_POINTS.flatMap((target, pointIndex) =>
      Array.from({ length: perPoint }, () => ({
        image: new Blob(['jpeg']), target, pointIndex,
        viewportWidth: 800, viewportHeight: 600, timestamp: 100,
      })),
    ),
  }
}

function prediction(x = 0.5, y = 0.5): CalibrationImagePrediction {
  return {
    ok: true, prediction: { x, y }, image: { width: 640, height: 480 },
    intrinsics: { focalLengthPx: 640, principalPoint: { x: 320, y: 240 } },
  }
}

beforeEach(() => {
  vi.resetAllMocks()
  predict.mockResolvedValue(prediction())
})

test('fits all valid raw pairs and retains samples, outcomes, counts, and progress', async () => {
  const captured = capture()
  captured.samples.forEach(({ target }) => predict.mockResolvedValueOnce(prediction(target.x, target.y)))
  const onProgress = vi.fn()
  const result = await processCalibration(captured, { onProgress })
  expect(result.ok).toBe(true)
  if (!result.ok) throw new Error(result.reason)
  expect(result.calibration.capture).toBe(captured)
  expect(result.calibration.frames).toHaveLength(27)
  expect(result.calibration.validFrameCounts).toEqual(Array(9).fill(3))
  result.calibration.frames.forEach((frame, i) => {
    expect(frame.sample).toBe(captured.samples[i])
    expect(frame.result).toEqual(prediction(frame.sample.target.x, frame.sample.target.y))
    expect(predict).toHaveBeenNthCalledWith(i + 1, frame.sample.image)
  })
  expect(result.calibration.fit.lambda).toBe(0.1)
  // Nine-point grid, three repeats: centered variance on each axis is 2.88.
  expect(result.calibration.fit.weight[0][0]).toBeCloseTo(2.88 / 2.98, 12)
  expect(result.calibration.fit.weight[1][1]).toBeCloseTo(2.88 / 2.98, 12)
  expect(result.calibration.fit.weight[0][1]).toBeCloseTo(0, 12)
  expect(onProgress.mock.calls.map(([progress]) => progress.completed)).toEqual(
    Array.from({ length: 28 }, (_, i) => i),
  )
  expect(onProgress).toHaveBeenLastCalledWith({ completed: 27, total: 27 })
})

test('awaits each prediction before starting the next frame', async () => {
  let finish!: (value: CalibrationImagePrediction) => void
  predict.mockReturnValueOnce(new Promise((resolve) => { finish = resolve }))
  const pending = processCalibration(capture(1))
  expect(predict).toHaveBeenCalledTimes(1)
  finish(prediction())
  expect((await pending).ok).toBe(true)
  expect(predict).toHaveBeenCalledTimes(9)
})

test('keeps a rejected frame and fits the remaining frames when coverage survives', async () => {
  predict.mockResolvedValueOnce({ ok: false, stage: 'preprocess', reason: 'face_not_detected' })
  const result = await processCalibration(capture())
  expect(result.ok).toBe(true)
  if (!result.ok) throw new Error(result.reason)
  expect(result.calibration.validFrameCounts).toEqual([2, 3, 3, 3, 3, 3, 3, 3, 3])
  expect(result.calibration.frames[0].result).toEqual({
    ok: false, stage: 'preprocess', reason: 'face_not_detected',
  })
})

test('reports missing coverage when every frame at a target fails', async () => {
  for (let i = 0; i < 3; i++) predict.mockResolvedValueOnce({ ok: false, stage: 'preprocess', reason: 'pose_failed' })
  const result = await processCalibration(capture())
  expect(result).toMatchObject({ ok: false, missingPointIndices: [0] })
  if (result.ok) throw new Error('Expected missing coverage')
  expect(result.reason).toMatch(/target.*1.*pose_failed/i)
  expect(result.frames).toHaveLength(27)
  expect(result).not.toHaveProperty('calibration')
})

test('requires nine distinct targets rather than nine valid frames', async () => {
  const captured = capture()
  const result = await processCalibration({ ...captured, samples: captured.samples.slice(0, 24) })
  expect(result).toMatchObject({ ok: false, missingPointIndices: [8] })
})

test('rejects an empty capture without calling inference', async () => {
  expect(await processCalibration({ samples: [], completedAt: 0 })).toMatchObject({ ok: false })
  expect(predict).not.toHaveBeenCalled()
})

test.each([
  { pointIndex: 9, target: CALIBRATION_POINTS[0] },
  { pointIndex: 0.5, target: CALIBRATION_POINTS[0] },
  { pointIndex: 0, target: CALIBRATION_POINTS[1] },
  { pointIndex: 0, target: { x: NaN, y: 0.1 } },
])('rejects incorrect target metadata: %j', async (metadata) => {
  const captured = capture()
  const samples = [...captured.samples]
  samples[0] = { ...samples[0], ...metadata }
  const result = await processCalibration({ ...captured, samples })
  expect(result).toMatchObject({ ok: false, reason: expect.stringMatching(/target metadata/i) })
  expect(predict).not.toHaveBeenCalled()
})

test('retains unexpected prediction failures with their sample', async () => {
  predict.mockRejectedValueOnce(new Error('runtime failed'))
  const result = await processCalibration(capture(1))
  expect(result.ok).toBe(false)
  if (result.ok) throw new Error('Expected failure')
  expect(result.frames[0].result).toEqual({ ok: false, stage: 'inference', reason: 'runtime failed' })
})

test('does not publish a fit when fitting overflows', async () => {
  predict.mockResolvedValue(prediction(1e308, -1e308))
  predict.mockResolvedValueOnce(prediction(-1e308, 1e308))
  const result = await processCalibration(capture(1))
  expect(result).toMatchObject({ ok: false, reason: expect.stringMatching(/fitting failed.*non-finite/i) })
  expect(result).not.toHaveProperty('calibration')
})

test('rejects a non-finite prediction as a failed frame', async () => {
  predict.mockResolvedValueOnce(prediction(NaN, 0.5))
  const result = await processCalibration(capture(1))
  expect(result.ok).toBe(false)
  if (result.ok) throw new Error('Expected failure')
  expect(result.frames[0].result).toMatchObject({ ok: false, reason: 'non_finite_prediction' })
})

test('does not start an already cancelled run', async () => {
  const controller = new AbortController()
  controller.abort()
  await expect(processCalibration(capture(), { signal: controller.signal })).rejects.toMatchObject({ name: 'AbortError' })
  expect(predict).not.toHaveBeenCalled()
})

test('stops after an in-flight prediction resolves when cancelled', async () => {
  let finish!: (value: CalibrationImagePrediction) => void
  predict.mockReturnValueOnce(new Promise((resolve) => { finish = resolve }))
  const controller = new AbortController()
  const onProgress = vi.fn()
  const pending = processCalibration(capture(), { signal: controller.signal, onProgress })
  const rejected = expect(pending).rejects.toMatchObject({ name: 'AbortError' })
  controller.abort()
  finish(prediction())
  await rejected
  expect(predict).toHaveBeenCalledTimes(1)
  expect(onProgress).toHaveBeenCalledTimes(1)
})
