import { calibrationCompatibilityMatches, type CalibrationSession } from './record'

export interface CaptureCamera {
  readonly deviceId: string
  readonly label: string
  readonly width: number
  readonly height: number
  readonly facingMode: string | null
}

export function camerasMatch(expected: CaptureCamera | null | undefined, actual: CaptureCamera | null): boolean {
  return !!expected && !!actual && expected.deviceId === actual.deviceId &&
    expected.width === actual.width && expected.height === actual.height && expected.facingMode === actual.facingMode
}

export function calibrationMatchesFrame(
  session: CalibrationSession | null | undefined,
  actual: CaptureCamera | null,
  image: { cols: number; rows: number },
): boolean {
  const expected = session?.record.calibration.capture.camera
  return session?.cameraVerification === 'verified' && session.record.version === 1 &&
    calibrationCompatibilityMatches(session.record.compatibility) && camerasMatch(expected, actual) &&
    image.cols === expected?.width && image.rows === expected?.height
}

export const CAMERA_CONSTRAINTS: MediaStreamConstraints = {
  video: { facingMode: 'user' }, audio: false,
}

export function captureCameraSettings(stream: MediaStream | null): CaptureCamera | null {
  const track = stream?.getVideoTracks?.()[0]
  const settings = track?.getSettings?.()
  if (!settings?.deviceId || !Number.isSafeInteger(settings.width) || !Number.isSafeInteger(settings.height) ||
      settings.width! <= 0 || settings.height! <= 0) return null
  return {
    deviceId: settings.deviceId, label: track?.label ?? '',
    width: settings.width!, height: settings.height!, facingMode: settings.facingMode ?? null,
  }
}

/** Request camera verification only on a user action; stop tracks on every exit. */
export async function verifyCaptureCamera(expected: CaptureCamera, signal?: AbortSignal): Promise<void> {
  signal?.throwIfAborted()
  if (!navigator.mediaDevices?.getUserMedia) throw new Error('Camera cannot be verified. Recalibrate on a supported browser.')
  const stream = await navigator.mediaDevices.getUserMedia(CAMERA_CONSTRAINTS)
  try {
    signal?.throwIfAborted()
    const actual = captureCameraSettings(stream)
    if (!camerasMatch(expected, actual)) {
      throw new Error('Camera identity or image dimensions changed or are unavailable. Recalibrate before reuse.')
    }
  } finally {
    stream.getTracks().forEach((track) => track.stop())
  }
}
