/**
 * Turn a calibration run into a per-user gaze correction.
 *
 * Calibration records a few webcam frames at each of nine targets. Every
 * frame goes through the same model the maze uses, the predictions for one
 * target are averaged (cutting per-target noise by sqrt(frames)), and a ridge
 * fit maps those averages onto the targets.
 *
 * What this removes is per-user bias: a gaze that reads compressed toward the
 * center, or shifted. It does not remove frame-to-frame noise, which on this
 * model is the larger share of the error; that needs smoothing over time.
 *
 * The fit lives for the page session, with the CalibrationResult it came
 * from. Persisting it across reloads is a separate concern (issue #49).
 */

import type { CalibrationResult } from './points'
import { applyRidge, fitRidge, type RidgeFit, type ScreenPoint } from './ridge'

/**
 * The penalty's strength is relative to the spread of the data, so it is not
 * automatically the right value for a different grid. It was checked for this
 * one (targets at 0.1/0.5/0.9, three frames averaged) in simulation: at the
 * model's measured per-frame noise (sigma ~0.11 screen fractions) 0.1 was the
 * best of 0 / 0.003 / 0.01 / 0.03 / 0.1 / 0.3, and within 7% of the best down
 * to sigma 0.06. It also matches the value model/ settled on offline.
 */
export const RIDGE_LAMBDA = 0.1

/** Usable targets needed before a fit is trusted (the map has 6 parameters). */
export const MIN_POINTS = 5

/**
 * Below this RMS spread of the averaged predictions (in screen fractions) the
 * predictions do not follow the targets at all: the user never moved their
 * eyes, or the camera saw a still image. A fit on that collapses every gaze
 * onto one point and freezes the cursor, which is worse than no correction.
 */
export const MIN_PREDICTION_SPREAD = 0.02

/** A frame's prediction, or null when preprocessing dropped the frame. */
export type FramePredictor = (frame: Blob) => Promise<ScreenPoint | null>

export type CalibrationFit =
  | {
      status: 'ok'
      fit: RidgeFit
      /** Targets with at least one usable frame. */
      points: number
      droppedFrames: number
      /** Mean distance from target before correction, in screen fractions. */
      rawError: number
      /** Leave-one-target-out error after correction: an honest estimate. */
      looError: number
    }
  | {
      status: 'failed'
      reason: 'too_few_points' | 'flat_predictions'
      points: number
      droppedFrames: number
    }

const distance = (a: ScreenPoint, b: ScreenPoint) => Math.hypot(a.x - b.x, a.y - b.y)

export async function fitCalibration(
  calibration: CalibrationResult,
  predict: FramePredictor,
  lamb = RIDGE_LAMBDA,
): Promise<CalibrationFit> {
  // Group usable predictions by target, in capture order.
  interface Target {
    target: ScreenPoint
    predictions: ScreenPoint[]
  }
  const byTarget = new Map<number, Target>()
  let droppedFrames = 0
  for (const sample of calibration.samples) {
    const prediction = await predict(sample.image)
    if (prediction === null) {
      droppedFrames++
      continue
    }
    const entry: Target = byTarget.get(sample.pointIndex) ?? { target: sample.target, predictions: [] }
    entry.predictions.push(prediction)
    byTarget.set(sample.pointIndex, entry)
  }

  const targets: ScreenPoint[] = []
  const means: ScreenPoint[] = []
  for (const { target, predictions } of byTarget.values()) {
    targets.push({ x: target.x, y: target.y })
    means.push({
      x: predictions.reduce((s, p) => s + p.x, 0) / predictions.length,
      y: predictions.reduce((s, p) => s + p.y, 0) / predictions.length,
    })
  }
  const points = targets.length
  if (points < MIN_POINTS) return { status: 'failed', reason: 'too_few_points', points, droppedFrames }

  const cx = means.reduce((s, p) => s + p.x, 0) / points
  const cy = means.reduce((s, p) => s + p.y, 0) / points
  const spread = Math.sqrt(means.reduce((s, p) => s + (p.x - cx) ** 2 + (p.y - cy) ** 2, 0) / points)
  if (spread < MIN_PREDICTION_SPREAD) {
    return { status: 'failed', reason: 'flat_predictions', points, droppedFrames }
  }

  const fit = fitRidge(means, targets, lamb)
  const rawError = means.reduce((s, p, i) => s + distance(p, targets[i]), 0) / points
  let looError = 0
  for (let i = 0; i < points; i++) {
    const held = fitRidge(
      means.filter((_, j) => j !== i),
      targets.filter((_, j) => j !== i),
      lamb,
    )
    looError += distance(applyRidge(held, means[i]), targets[i]) / points
  }
  return { status: 'ok', fit, points, droppedFrames, rawError, looError }
}

/**
 * Correct one prediction and clamp it to the screen. The affine map can push
 * a gaze past an edge, and a look past the right edge still means "right".
 */
export function correctGaze(fit: RidgeFit | null, p: ScreenPoint): ScreenPoint {
  const out = fit ? applyRidge(fit, p) : p
  return { x: Math.min(1, Math.max(0, out.x)), y: Math.min(1, Math.max(0, out.y)) }
}
