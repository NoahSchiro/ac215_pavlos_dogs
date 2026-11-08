import type { CameraIntrinsics } from './utils'

/**
 * Webcam approximation shared with PR #68's live inference loop.
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
