import { camerasMatch, type CaptureCamera } from '../vision/camera'
import { calibrationCompatibilityMatches, type CalibrationSession } from './record'

export function calibrationMatchesFrame(
  session: CalibrationSession | null | undefined,
  actual: CaptureCamera | null,
  image: { cols: number; rows: number },
): boolean {
  const expected = session?.record.calibration.camera
  return session?.cameraVerification === 'verified' && session.record.version === 2 &&
    calibrationCompatibilityMatches(session.record.compatibility) && camerasMatch(expected, actual) &&
    image.cols === expected?.width && image.rows === expected?.height
}
