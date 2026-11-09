import { predictCalibrationImage, type CalibrationImagePrediction } from './inference'
import { CALIBRATION_POINTS, type CalibrationResult, type CalibrationSample, type NormalizedPoint } from './points'
import { fitCalibration, type CalibrationFit } from './ridge'

export interface EvaluatedSample {
  readonly sample: CalibrationSample
  readonly result: CalibrationImagePrediction
}

export interface PersonalCalibration {
  readonly capture: CalibrationResult
  readonly frames: readonly EvaluatedSample[]
  readonly validFrameCounts: readonly number[]
  readonly fit: CalibrationFit
}

export interface CalibrationProgress {
  readonly completed: number
  readonly total: number
}

export type CalibrationTaskState =
  | { status: 'idle' | 'cancelled' | 'restoring' | 'restored' | 'saving' | 'clearing' | 'cleared' | 'verifying' | 'verified' }
  | ({ status: 'processing' } & CalibrationProgress)
  | { status: 'success'; validFrames: number; total: number }
  | { status: 'error'; message: string }

export type CalibrationProcessingResult =
  | { ok: true; calibration: PersonalCalibration }
  | { ok: false; reason: string; frames: readonly EvaluatedSample[]; missingPointIndices: readonly number[] }

// Workflow quality policy, separate from ridge's valid constant/collinear fits.
// Require at least 1% screen-coordinate standard deviation in every direction.
const MIN_TARGET_VARIANCE = 0.01 ** 2

function hasSufficientVariation(frames: readonly EvaluatedSample[], counts: readonly number[]): boolean {
  // Average each target's burst first so within-burst noise cannot supply variation.
  const means = CALIBRATION_POINTS.map(() => ({ x: 0, y: 0 }))
  for (const { sample, result } of frames) {
    if (!result.ok) continue
    means[sample.pointIndex].x += result.prediction.x / counts[sample.pointIndex]
    means[sample.pointIndex].y += result.prediction.y / counts[sample.pointIndex]
  }
  const n = means.length
  const center = means.reduce((sum, point) => ({ x: sum.x + point.x / n, y: sum.y + point.y / n }), { x: 0, y: 0 })
  let xx = 0, xy = 0, yy = 0
  for (const point of means) {
    const dx = point.x - center.x, dy = point.y - center.y
    xx += dx * dx / n
    xy += dx * dy / n
    yy += dy * dy / n
  }
  // The smaller covariance eigenvalue catches collapsed axes and diagonal lines.
  const variance = (xx + yy - Math.hypot(xx - yy, 2 * xy)) / 2
  return Number.isFinite(variance) && variance >= MIN_TARGET_VARIANCE
}

export async function processCalibration(
  capture: CalibrationResult,
  options: { signal?: AbortSignal; onProgress?: (progress: CalibrationProgress) => void } = {},
): Promise<CalibrationProcessingResult> {
  const { signal, onProgress } = options
  signal?.throwIfAborted()
  const frames: EvaluatedSample[] = []
  const failure = (reason: string, missingPointIndices: number[] = []): CalibrationProcessingResult =>
    ({ ok: false, reason, frames, missingPointIndices })

  if (capture.samples.length === 0) {
    return failure('No captured frames. Capture all nine targets again.', CALIBRATION_POINTS.map((_, i) => i))
  }
  const invalidIndex = capture.samples.findIndex((sample) => {
    const target = CALIBRATION_POINTS[sample.pointIndex]
    return !Number.isInteger(sample.pointIndex) || !target || !sample.target ||
      sample.target.x !== target.x || sample.target.y !== target.y
  })
  if (invalidIndex !== -1) return failure(`Invalid target metadata for sample ${invalidIndex + 1}. Capture again.`)

  const validFrameCounts = CALIBRATION_POINTS.map(() => 0)
  const predictions: NormalizedPoint[] = []
  const targets: NormalizedPoint[] = []
  onProgress?.({ completed: 0, total: capture.samples.length })
  for (const sample of capture.samples) {
    signal?.throwIfAborted()
    let result: CalibrationImagePrediction
    try {
      result = await predictCalibrationImage(sample.image)
    } catch (error) {
      result = { ok: false, stage: 'inference', reason: error instanceof Error ? error.message : String(error) }
    }
    signal?.throwIfAborted()
    if (result.ok && (!Number.isFinite(result.prediction.x) || !Number.isFinite(result.prediction.y))) {
      result = { ok: false, stage: 'inference', reason: 'non_finite_prediction' }
    }
    frames.push({ sample, result })
    if (result.ok) {
      validFrameCounts[sample.pointIndex]++
      predictions.push(result.prediction)
      targets.push(sample.target)
    }
    onProgress?.({ completed: frames.length, total: capture.samples.length })
  }
  signal?.throwIfAborted()
  const missing = validFrameCounts.flatMap((count, index) => count === 0 ? [index] : [])
  if (missing.length) {
    const rejected = frames.map((frame) => frame.result).find((result) => !result.ok)
    const detail = rejected && !rejected.ok ? ` First frame failure: ${rejected.stage}: ${rejected.reason}.` : ''
    return failure(`No valid frame at target(s) ${missing.map((i) => i + 1).join(', ')}. Capture again.${detail}`, missing)
  }
  if (!hasSufficientVariation(frames, validFrameCounts)) {
    return failure('Insufficient prediction variation across targets. Look at each target and capture again.')
  }
  try {
    const fit = fitCalibration(predictions, targets)
    return { ok: true, calibration: { capture, frames, validFrameCounts, fit } }
  } catch (error) {
    return failure(`Calibration fitting failed: ${error instanceof Error ? error.message : String(error)}`)
  }
}
