import { FRAMES_PER_POINT, VALIDATION_POINTS, type CalibrationResult } from './points'
import type { CalibrationSession } from './record'
import { calibrationMatchesFrame } from './cameraMetadata'
import { predictCalibrationImage } from './inference'
import { applyCalibration } from './ridge'

export interface ValidationSummary {
  rawMeanError: number
  correctedMeanError: number
  validFrameCount: number
}

export async function validateCalibrationCapture(
  capture: CalibrationResult,
  session: CalibrationSession,
  options: { signal?: AbortSignal; onProgress?: (completed: number) => void } = {},
): Promise<ValidationSummary> {
  const { signal, onProgress } = options
  signal?.throwIfAborted()
  const camera = capture.camera ?? null
  if (!camera || !calibrationMatchesFrame(session, camera, { cols: camera.width, rows: camera.height })) {
    throw new Error('Verify the matching camera or recalibrate before checking accuracy.')
  }
  const training = session.record.calibration.capture
  if (capture.samples.length !== VALIDATION_POINTS.length * FRAMES_PER_POINT ||
      !Number.isFinite(capture.completedAt) || capture.completedAt <= training.completedAt) {
    throw new Error('Capture 12 fresh frames at the four validation targets.')
  }
  const capturedCounts = VALIDATION_POINTS.map(() => 0)
  for (const sample of capture.samples) {
    const target = VALIDATION_POINTS[sample.pointIndex]
    if (!Number.isInteger(sample.pointIndex) || !target || sample.target.x !== target.x || sample.target.y !== target.y ||
        !Number.isFinite(sample.timestamp) || sample.timestamp <= training.completedAt || sample.timestamp > capture.completedAt ||
        training.samples.some(({ image }) => image === sample.image)) {
      throw new Error('Use fresh images at the fixed validation targets, not fitting frames.')
    }
    capturedCounts[sample.pointIndex]++
  }
  if (capturedCounts.some((count) => count !== FRAMES_PER_POINT)) throw new Error('Capture three frames per validation target.')
  const validCounts = VALIDATION_POINTS.map(() => 0)
  let rawError = 0, correctedError = 0
  for (const [index, sample] of capture.samples.entries()) {
    signal?.throwIfAborted()
    const result = await predictCalibrationImage(sample.image)
    signal?.throwIfAborted()
    if (result.ok) {
      if (!calibrationMatchesFrame(session, camera, { cols: result.image.width, rows: result.image.height })) {
        throw new Error('Camera image dimensions changed. Recalibrate before checking accuracy.')
      }
      const raw = result.prediction
      const corrected = applyCalibration(raw, session.record.calibration.fit)
      rawError += Math.hypot(raw.x - sample.target.x, raw.y - sample.target.y)
      correctedError += Math.hypot(corrected.x - sample.target.x, corrected.y - sample.target.y)
      validCounts[sample.pointIndex]++
    }
    onProgress?.(index + 1)
  }
  signal?.throwIfAborted()
  const missing = validCounts.flatMap((count, index) => count === 0 ? [index + 1] : [])
  if (missing.length) throw new Error(`No valid frame at target(s) ${missing.join(', ')}. Try again.`)
  if (!Number.isFinite(rawError) || !Number.isFinite(correctedError)) throw new Error('Accuracy calculation is not finite.')
  const validFrameCount = validCounts.reduce((total, count) => total + count, 0)
  return { rawMeanError: rawError / validFrameCount, correctedMeanError: correctedError / validFrameCount, validFrameCount }
}
