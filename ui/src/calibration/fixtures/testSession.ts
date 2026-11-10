import type { CaptureCamera } from '../../vision/camera'
import { CALIBRATION_POINTS } from '../points'
import { createCalibrationRecord, type CalibrationSession } from '../record'
import { fitCalibration } from '../ridge'

export const TEST_CAMERA: CaptureCamera = {
  deviceId: 'camera-a', label: 'Camera A', width: 640, height: 480, facingMode: 'user',
}

/** An exact horizontal mirror makes applying the correction twice observable. */
export function testSession(): CalibrationSession {
  const samples = CALIBRATION_POINTS.map((target, pointIndex) => ({
    image: new Blob(['jpeg'], { type: 'image/jpeg' }), target, pointIndex,
    timestamp: 1, viewportWidth: 800, viewportHeight: 600,
  }))
  const predictions = samples.map(({ target }) => ({ x: 1 - target.x, y: target.y }))
  return {
    cameraVerification: 'verified',
    record: createCalibrationRecord({
      capture: { samples, camera: TEST_CAMERA, completedAt: 2 },
      validFrameCounts: Array(9).fill(1),
      fit: fitCalibration(predictions, CALIBRATION_POINTS, 0),
      frames: samples.map((sample, i) => ({ sample, result: {
        ok: true, prediction: predictions[i], image: { width: 640, height: 480 },
        intrinsics: { focalLengthPx: 640, principalPoint: { x: 320, y: 240 } },
      } })),
    }),
  }
}
