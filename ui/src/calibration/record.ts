import type { PersonalCalibration } from './process'
import { CALIBRATION_POINTS, FRAMES_PER_POINT } from './points'
import { applyCalibration, type CalibrationFit } from './ridge'
import type { CaptureCamera } from '../vision/camera'
import type { Viewport } from '../vision/viewport'
import { captureViewport } from './viewport'

// A saved correction depends on the model outputs used during calibration.
export const CALIBRATION_COMPATIBILITY = {
  modelId: 'sha256:c4a11ab96dc64766cd1abd622838b50b297aaec4adf4f25ebf71cf0b66bcbee6',
  preprocessingVersion: 1,
  intrinsicsVersion: 1,
  imageOrientation: 'unmirrored',
} as const

export interface CalibrationRecord {
  readonly version: 2
  readonly savedAt: number
  readonly compatibility: typeof CALIBRATION_COMPATIBILITY
  readonly calibration: {
    readonly fit: CalibrationFit
    readonly camera: CaptureCamera
    readonly viewport: Viewport
    readonly completedAt: number
    readonly validFrameCounts: readonly number[]
  }
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
  const { capture, fit, validFrameCounts, frames } = calibration
  const viewport = captureViewport(capture)
  requireValid(viewport, 'Calibration contains inconsistent viewport sizes. Recalibrate.')
  const camera = capture.camera
  requireValid(camera, 'Capture camera identity is unavailable. Recalibrate.')
  requireValid(frames.every(({ result }) => !result.ok ||
    (result.image.width === camera.width && result.image.height === camera.height)),
  'Image dimensions do not match capture camera. Recalibrate.')
  const record: CalibrationRecord = {
    version: 2, savedAt: Date.now(), compatibility: { ...CALIBRATION_COMPATIBILITY },
    calibration: { fit, camera: { ...camera }, viewport, completedAt: capture.completedAt,
      validFrameCounts: [...validFrameCounts] },
  }
  validateCalibrationRecord(record)
  return record
}

function requireValid(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message)
}
const finite = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value)
const positiveInteger = (value: number) => Number.isSafeInteger(value) && value > 0

/** Validate the saved correction and metadata without retaining fitting frames. */
export function validateCalibrationRecord(value: unknown): asserts value is CalibrationRecord {
  const r = value as CalibrationRecord
  requireValid(r && r.version === 2, 'Unsupported calibration record version. Recalibrate.')
  requireValid(calibrationCompatibilityMatches(r.compatibility), 'Model or preprocessing settings changed. Recalibrate.')
  requireValid(finite(r.savedAt) && r.savedAt >= 0, 'Invalid calibration save time.')
  const calibration = r.calibration
  requireValid(calibration && finite(calibration.completedAt) && calibration.completedAt >= 0 &&
    calibration.completedAt <= r.savedAt, 'Invalid calibration capture time.')
  const camera = calibration.camera
  requireValid(camera && typeof camera.deviceId === 'string' && camera.deviceId.trim() &&
    typeof camera.label === 'string' && positiveInteger(camera.width) && positiveInteger(camera.height) &&
    (camera.facingMode === null || typeof camera.facingMode === 'string'),
  'Capture camera identity or dimensions are unavailable. Recalibrate on a supported camera.')
  requireValid(calibration.viewport && positiveInteger(calibration.viewport.width) && positiveInteger(calibration.viewport.height),
    'Invalid calibration viewport size. Recalibrate.')
  requireValid(Array.isArray(calibration.validFrameCounts) && calibration.validFrameCounts.length === CALIBRATION_POINTS.length &&
    calibration.validFrameCounts.every((count) => Number.isInteger(count) && count > 0 && count <= FRAMES_PER_POINT),
    'Invalid calibration frame counts.')
  requireValid(calibration.fit && finite(calibration.fit.lambda) && calibration.fit.lambda >= 0, 'Invalid saved ridge lambda.')
  applyCalibration({ x: 0.5, y: 0.5 }, calibration.fit)
}
