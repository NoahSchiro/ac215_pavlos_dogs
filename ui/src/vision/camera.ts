import type { CameraIntrinsics } from './utils'

export interface CaptureCamera {
  readonly deviceId: string
  readonly label: string
  readonly width: number
  readonly height: number
  readonly facingMode: string | null
}

export const CAMERA_CONSTRAINTS: MediaStreamConstraints = {
  video: { facingMode: 'user' }, audio: false,
}

export function camerasMatch(expected: CaptureCamera | null | undefined, actual: CaptureCamera | null): boolean {
  return !!expected && !!actual && expected.deviceId === actual.deviceId &&
    expected.width === actual.width && expected.height === actual.height && expected.facingMode === actual.facingMode
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

/**
 * Approximate webcam intrinsics from the actual frame dimensions.
 * Pass actual image dimensions, never viewport or CSS preview dimensions.
 */
export function approximateCameraIntrinsics(width: number, height: number): CameraIntrinsics {
  if (!Number.isSafeInteger(width) || !Number.isSafeInteger(height) || width <= 0 || height <= 0) {
    throw new Error('Image dimensions must be positive integers')
  }
  return {
    focalLengthPx: Math.max(width, height),
    principalPoint: { x: width / 2, y: height / 2 },
  }
}
