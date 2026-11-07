import { describe, expect, it } from 'vitest'
import { correctGaze, fitCalibration, MIN_POINTS, type FramePredictor } from './fit'
import { CALIBRATION_POINTS, type CalibrationResult, type CalibrationSample } from './points'
import type { ScreenPoint } from './ridge'

/** One fake frame per (target, repeat); `predict` looks the prediction up. */
function run(
  predictionFor: (target: ScreenPoint, repeat: number, index: number) => ScreenPoint | null,
  frames = 3,
) {
  const table = new Map<Blob, ScreenPoint | null>()
  const samples: CalibrationSample[] = []
  CALIBRATION_POINTS.forEach((target, pointIndex) => {
    for (let r = 0; r < frames; r++) {
      const image = new Blob([`${pointIndex}-${r}`])
      table.set(image, predictionFor(target, r, pointIndex))
      samples.push({ image, target, pointIndex, viewportWidth: 1280, viewportHeight: 720, timestamp: 0 })
    }
  })
  const result: CalibrationResult = { samples, completedAt: 0 }
  const predict: FramePredictor = async (blob) => table.get(blob) ?? null
  return { result, predict }
}

// A model that is compressed toward the center and shifted: what a raw,
// uncalibrated webcam prediction tends to look like.
const biased = (t: ScreenPoint): ScreenPoint => ({ x: 0.6 * t.x + 0.25, y: 0.7 * t.y + 0.05 })

describe('fitCalibration', () => {
  it('learns a correction that beats the raw predictions on held-out targets', async () => {
    const { result, predict } = run((t) => biased(t))
    const fit = await fitCalibration(result, predict)
    expect(fit.status).toBe('ok')
    if (fit.status !== 'ok') return
    expect(fit.points).toBe(9)
    expect(fit.droppedFrames).toBe(0)
    // The penalty trades some correction for robustness to noise, so even a
    // noiseless bias is only partly removed -- but it must be removed.
    expect(fit.looError).toBeLessThan(0.8 * fit.rawError)
    const target = { x: 0.9, y: 0.5 }
    const raw = biased(target)
    const corrected = correctGaze(fit.fit, raw)
    expect(Math.abs(corrected.x - target.x)).toBeLessThan(Math.abs(raw.x - target.x))
  })

  it('averages the frames of a target before fitting', async () => {
    // Symmetric jitter around the biased prediction averages out exactly.
    const jitter = [-0.03, 0, 0.03]
    const noisy = run((t, r) => ({ x: biased(t).x + jitter[r], y: biased(t).y - jitter[r] }))
    const clean = run((t) => biased(t))
    const a = await fitCalibration(noisy.result, noisy.predict)
    const b = await fitCalibration(clean.result, clean.predict)
    if (a.status !== 'ok' || b.status !== 'ok') throw new Error('expected both fits to succeed')
    a.fit.weight.forEach((w, i) => expect(w).toBeCloseTo(b.fit.weight[i], 12))
  })

  it('skips dropped frames and still fits with enough targets', async () => {
    const { result, predict } = run((t, r, i) => (i < 3 || r === 0 ? null : biased(t)))
    const fit = await fitCalibration(result, predict)
    expect(fit.status).toBe('ok')
    expect(fit.points).toBe(6)
    expect(fit.droppedFrames).toBe(9 + 6)
  })

  it(`refuses to fit on fewer than ${MIN_POINTS} usable targets`, async () => {
    const { result, predict } = run((t, _r, i) => (i < 5 ? null : biased(t)))
    expect(await fitCalibration(result, predict)).toMatchObject({
      status: 'failed',
      reason: 'too_few_points',
      points: 4,
    })
  })

  it('refuses to fit when predictions do not follow the targets', async () => {
    // A still image (or a user who never moved their eyes) predicts one point
    // for every target; fitting that would pin the cursor in place.
    const { result, predict } = run(() => ({ x: 0.62, y: 0.41 }))
    expect(await fitCalibration(result, predict)).toMatchObject({
      status: 'failed',
      reason: 'flat_predictions',
    })
  })
})

describe('correctGaze', () => {
  it('passes predictions through unchanged without a fit', () => {
    expect(correctGaze(null, { x: 0.3, y: 0.7 })).toEqual({ x: 0.3, y: 0.7 })
  })

  it('clamps corrected gaze to the screen', () => {
    const fit = { weight: [2, 0, 0, 2] as const, intercept: [0, -0.5] as const }
    expect(correctGaze(fit, { x: 0.9, y: 0.1 })).toEqual({ x: 1, y: 0 })
  })
})
