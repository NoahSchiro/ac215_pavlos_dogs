import type { CalibrationProgress } from './process'
import type { CalibrationResult, NormalizedPoint } from './points'
import type { CalibrationSession } from './record'
import type { CalibrationFit } from './ridge'
import type { CaptureCamera } from './cameraMetadata'
import { calibrationMatchesFrame } from './cameraMetadata'
import { predictCalibrationImage } from './inference'
import { applyCalibration } from './ridge'

export const VALIDATION_POINTS: readonly NormalizedPoint[] = [
  { x: 0.25, y: 0.25 }, { x: 0.75, y: 0.25 },
  { x: 0.25, y: 0.75 }, { x: 0.75, y: 0.75 },
]

export interface ValidationFrame {
  readonly pointIndex: number
  readonly target: NormalizedPoint
  readonly timestamp: number
  readonly raw?: NormalizedPoint
  readonly corrected?: NormalizedPoint
  readonly failure?: string
}

export interface ValidationReport {
  readonly version: 1
  readonly metric: 'mean-euclidean-normalized-unclipped'
  readonly targets: readonly NormalizedPoint[]
  readonly plannedFrameCount: 12
  readonly validFrameCount: number
  readonly validFrameCounts: readonly number[]
  readonly rawMeanError: number
  readonly correctedMeanError: number
  readonly improved: boolean
  readonly capturedAt: number
  readonly evaluatedAt: number
  readonly fitSavedAt: number
  readonly compatibility: CalibrationSession['record']['compatibility']
  readonly fit: CalibrationFit
  readonly camera: CaptureCamera
  readonly frames: readonly ValidationFrame[]
}

export async function validateCalibrationCapture(
  capture: CalibrationResult,
  session: CalibrationSession,
  options: { signal?: AbortSignal; onProgress?: (progress: CalibrationProgress) => void } = {},
): Promise<ValidationReport> {
  const { signal, onProgress } = options
  signal?.throwIfAborted()
  const camera = capture.camera ?? null
  if (!camera || !calibrationMatchesFrame(session, camera, { cols: camera.width, rows: camera.height })) {
    throw new Error('Verify the matching saved camera or recalibrate before validation.')
  }
  const training = session.record.calibration.capture
  const trainingImages = new Set(training.samples.map((sample) => sample.image))
  const capturedCounts = VALIDATION_POINTS.map(() => 0)
  if (capture.samples.length !== 12 || !Number.isFinite(capture.completedAt) || capture.completedAt <= training.completedAt) {
    throw new Error('Validation requires 12 fresh frames captured after fitting.')
  }
  for (const sample of capture.samples) {
    const target = VALIDATION_POINTS[sample.pointIndex]
    if (!Number.isInteger(sample.pointIndex) || !target || sample.target.x !== target.x || sample.target.y !== target.y ||
        !Number.isFinite(sample.timestamp) || sample.timestamp <= training.completedAt || sample.timestamp > capture.completedAt ||
        trainingImages.has(sample.image)) {
      throw new Error('Use fresh captures at the fixed validation targets; fitting frames cannot be reused.')
    }
    capturedCounts[sample.pointIndex]++
  }
  if (capturedCounts.some((count) => count !== 3)) throw new Error('Capture three fresh frames at each validation target.')
  const validFrameCounts = VALIDATION_POINTS.map(() => 0)
  const frames: ValidationFrame[] = []
  const rawErrors: number[] = []
  const correctedErrors: number[] = []
  onProgress?.({ completed: 0, total: 12 })
  for (const sample of capture.samples) {
    signal?.throwIfAborted()
    const frame = { pointIndex: sample.pointIndex, target: sample.target, timestamp: sample.timestamp }
    try {
      const result = await predictCalibrationImage(sample.image)
      signal?.throwIfAborted()
      if (!result.ok) frames.push({ ...frame, failure: `${result.stage}: ${result.reason}` })
      else {
        if (!calibrationMatchesFrame(session, camera, { cols: result.image.width, rows: result.image.height })) {
          throw new Error('Validation image dimensions changed. Recalibrate before reuse.')
        }
        const raw = result.prediction
        // Primary accuracy uses unclipped correction, independent of gameplay.
        const corrected = applyCalibration(raw, session.record.calibration.fit)
        const rawError = Math.hypot(raw.x - sample.target.x, raw.y - sample.target.y)
        const correctedError = Math.hypot(corrected.x - sample.target.x, corrected.y - sample.target.y)
        if (!Number.isFinite(rawError) || !Number.isFinite(correctedError)) throw new Error('Non-finite validation error.')
        rawErrors.push(rawError)
        correctedErrors.push(correctedError)
        validFrameCounts[sample.pointIndex]++
        frames.push({ ...frame, raw, corrected })
      }
    } catch (error) {
      signal?.throwIfAborted()
      frames.push({ ...frame, failure: error instanceof Error ? error.message : String(error) })
    }
    onProgress?.({ completed: frames.length, total: 12 })
  }
  signal?.throwIfAborted()
  const missing = validFrameCounts.flatMap((count, i) => count === 0 ? [i + 1] : [])
  if (missing.length) throw new Error(`No valid paired frame at validation target(s) ${missing.join(', ')}. ${frames.find((frame) => frame.failure)?.failure ?? ''}`)
  const rawMeanError = rawErrors.reduce((sum, error) => sum + error / rawErrors.length, 0)
  const correctedMeanError = correctedErrors.reduce((sum, error) => sum + error / correctedErrors.length, 0)
  if (!Number.isFinite(rawMeanError) || !Number.isFinite(correctedMeanError)) throw new Error('Non-finite validation mean error.')
  return {
    version: 1, metric: 'mean-euclidean-normalized-unclipped', targets: VALIDATION_POINTS,
    plannedFrameCount: 12, validFrameCount: rawErrors.length, validFrameCounts,
    rawMeanError, correctedMeanError, improved: correctedMeanError < rawMeanError,
    capturedAt: capture.completedAt, evaluatedAt: Date.now(), fitSavedAt: session.record.savedAt,
    compatibility: session.record.compatibility, fit: session.record.calibration.fit, camera, frames,
  }
}
