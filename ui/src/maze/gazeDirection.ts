import { calibrationMatchesFrame } from '../calibration/compatibility'
import type { CaptureCamera } from '../vision/camera'
import type { CalibrationSession } from '../calibration/record'
import { applyCalibration } from '../calibration/ridge'
import type { ScreenFraction } from '../vision/model'

export type Direction = 'up' | 'down' | 'left' | 'right'

// Use the dominant axis outside the center dead zone; ties use the vertical axis.
export function gazeToDirection(x: number, y: number): Direction | null {
  if (!Number.isFinite(x) || !Number.isFinite(y)) return null
  if (x < 0 || x > 1 || y < 0 || y > 1) return null
  const dx = x < 0.4 ? x - 0.4 : x > 0.6 ? x - 0.6 : 0
  const dy = y < 0.4 ? y - 0.4 : y > 0.6 ? y - 0.6 : 0
  if (dx === 0 && dy === 0) return null
  if (Math.abs(dx) > Math.abs(dy)) return dx < 0 ? 'left' : 'right'
  return dy < 0 ? 'up' : 'down'
}

export function prepareGameplayGaze(
  raw: ScreenFraction,
  session: CalibrationSession | null | undefined,
  camera: CaptureCamera | null,
  image: { cols: number; rows: number },
): { point: ScreenFraction; calibrated: boolean } | null {
  if (!Number.isFinite(raw.x) || !Number.isFinite(raw.y)) return null
  if (!calibrationMatchesFrame(session, camera, image)) return { point: raw, calibrated: false }
  try {
    const point = applyCalibration(raw, session!.record.calibration.fit)
    if (!Number.isFinite(point.x) || !Number.isFinite(point.y)) return null
    return { calibrated: true, point: { x: Math.max(0, Math.min(1, point.x)), y: Math.max(0, Math.min(1, point.y)) } }
  } catch { return null }
}
