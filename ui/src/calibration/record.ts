import type { PersonalCalibration } from './process'
import { CALIBRATION_POINTS } from './points'
import { applyCalibration, fitCalibration } from './ridge'
import { approximateCameraIntrinsics } from '../vision/camera'

// Bump these identifiers whenever the pinned assets or preprocessing contract change.
export const CALIBRATION_COMPATIBILITY = {
  modelId: 'sha256:c4a11ab96dc64766cd1abd622838b50b297aaec4adf4f25ebf71cf0b66bcbee6',
  preprocessingVersion: 1,
  intrinsicsVersion: 1,
  imageOrientation: 'unmirrored',
} as const

export interface CalibrationRecord {
  readonly version: 1
  readonly savedAt: number
  readonly compatibility: typeof CALIBRATION_COMPATIBILITY
  readonly calibration: PersonalCalibration
}

export interface CalibrationSession {
  readonly record: CalibrationRecord
  readonly cameraVerification: 'pending' | 'verified'
}

export function calibrationCompatibilityMatches(value: unknown): boolean {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return false
  return Object.entries(CALIBRATION_COMPATIBILITY).every(([key, expected]) =>
    (value as Record<string, unknown>)[key] === expected,
  )
}

export function createCalibrationRecord(calibration: PersonalCalibration): CalibrationRecord {
  const record: CalibrationRecord = {
    version: 1, savedAt: Date.now(), compatibility: { ...CALIBRATION_COMPATIBILITY }, calibration,
  }
  validateCalibrationRecord(record)
  return record
}

function requireValid(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message)
}
const finite = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value)
const positiveInteger = (value: number) => Number.isSafeInteger(value) && value > 0

/** Validate both storage shape and the relationships needed to trust this fit. */
export function validateCalibrationRecord(value: unknown): asserts value is CalibrationRecord {
  const r = value as CalibrationRecord
  requireValid(r && r.version === 1, 'Unsupported calibration record version. Recalibrate.')
  requireValid(calibrationCompatibilityMatches(r.compatibility), 'Model or preprocessing settings changed. Recalibrate.')
  requireValid(finite(r.savedAt) && r.savedAt >= 0, 'Invalid calibration save time.')
  const calibration = r.calibration
  const capture = calibration?.capture
  requireValid(capture && finite(capture.completedAt) && capture.completedAt >= 0, 'Invalid calibration capture.')
  const camera = capture.camera
  requireValid(camera && typeof camera.deviceId === 'string' && camera.deviceId.trim() &&
    typeof camera.label === 'string' && positiveInteger(camera.width) && positiveInteger(camera.height) &&
    (camera.facingMode === null || typeof camera.facingMode === 'string'),
  'Capture camera identity or dimensions are unavailable. Recalibrate on a supported camera.')
  requireValid(Array.isArray(capture.samples) && capture.samples.length >= 9 && capture.samples.length <= 27,
    'Invalid calibration samples.')
  requireValid(Array.isArray(calibration.frames) && calibration.frames.length === capture.samples.length,
    'Invalid calibration frame outcomes.')
  requireValid(Array.isArray(calibration.validFrameCounts) && calibration.validFrameCounts.length === 9,
    'Invalid calibration frame counts.')
  const counts = CALIBRATION_POINTS.map(() => 0)
  const capturedCounts = CALIBRATION_POINTS.map(() => 0)
  const predictions = []
  const targets = []
  const expectedIntrinsics = approximateCameraIntrinsics(camera.width, camera.height)
  for (let i = 0; i < capture.samples.length; i++) {
    const sample = capture.samples[i]
    const target = CALIBRATION_POINTS[sample?.pointIndex]
    requireValid(sample && Number.isInteger(sample.pointIndex) && target &&
      sample.target?.x === target.x && sample.target?.y === target.y, 'Invalid sample target metadata.')
    requireValid(sample.image instanceof Blob && sample.image.size > 0 && sample.image.type === 'image/jpeg', 'Invalid sample JPEG Blob.')
    requireValid(positiveInteger(sample.viewportWidth) && positiveInteger(sample.viewportHeight) &&
      finite(sample.timestamp) && sample.timestamp >= 0 && sample.timestamp <= capture.completedAt, 'Invalid sample capture metadata.')
    capturedCounts[sample.pointIndex]++
    const frame = calibration.frames[i]
    requireValid(frame?.sample === sample && frame.result && typeof frame.result.ok === 'boolean', 'Invalid frame-to-sample association.')
    const result = frame.result
    if (result.ok) {
      requireValid(finite(result.prediction?.x) && finite(result.prediction?.y), 'Invalid raw prediction.')
      requireValid(result.image?.width === camera.width && result.image?.height === camera.height,
        'Image dimensions do not match capture camera. Recalibrate.')
      requireValid(result.intrinsics?.focalLengthPx === expectedIntrinsics.focalLengthPx &&
        result.intrinsics.principalPoint?.x === expectedIntrinsics.principalPoint.x &&
        result.intrinsics.principalPoint?.y === expectedIntrinsics.principalPoint.y, 'Invalid frame camera parameters.')
      counts[sample.pointIndex]++
      predictions.push(result.prediction)
      targets.push(sample.target)
    } else {
      requireValid(['decode', 'landmarker', 'model', 'preprocess', 'inference'].includes(result.stage) &&
        typeof result.reason === 'string' && result.reason.length > 0, 'Invalid failed-frame outcome.')
    }
  }
  requireValid(counts.every((count, i) => count > 0 && count === calibration.validFrameCounts[i] && capturedCounts[i] <= 3),
    'Calibration target coverage or valid-frame counts are inconsistent.')
  requireValid(calibration.fit && finite(calibration.fit.lambda) && calibration.fit.lambda >= 0, 'Invalid saved ridge lambda.')
  applyCalibration({ x: 0.5, y: 0.5 }, calibration.fit)
  const expectedFit = fitCalibration(predictions, targets, calibration.fit.lambda)
  const actual = [...calibration.fit.weight.flat(), ...calibration.fit.intercept]
  const expected = [...expectedFit.weight.flat(), ...expectedFit.intercept]
  requireValid(actual.every((coefficient, i) => Math.abs(coefficient - expected[i]) <= 1e-10), 'Stored fit does not match its samples.')
}
