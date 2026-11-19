import { approximateCameraIntrinsics } from '../vision/camera'
import { loadGazeModel, predictGaze, type ScreenFraction } from '../vision/model'
import { getLandmarker, type CameraIntrinsics } from '../vision/utils'
import { decodeJpeg } from '../vision/jpeg'

type FailureStage = 'decode' | 'landmarker' | 'model' | 'preprocess' | 'inference'

export type CalibrationImagePrediction =
  | {
      ok: true
      prediction: ScreenFraction
      image: { width: number; height: number }
      intrinsics: CameraIntrinsics
    }
  | { ok: false; stage: FailureStage; reason: string }

/** Decode a captured image and run the existing preprocessing and gaze model. */
export async function predictCalibrationImage(blob: Blob): Promise<CalibrationImagePrediction> {
  let stage: FailureStage = 'decode'
  try {
    const image = await decodeJpeg(blob)
    const intrinsics = approximateCameraIntrinsics(image.cols, image.rows)
    stage = 'landmarker'
    const landmarker = await getLandmarker()
    stage = 'model'
    const model = await loadGazeModel()
    stage = 'inference'
    const prediction = await predictGaze(landmarker, model, image, intrinsics)
    if (typeof prediction === 'string') {
      return { ok: false, stage: 'preprocess', reason: prediction }
    }
    if (!Number.isFinite(prediction.x) || !Number.isFinite(prediction.y)) {
      return { ok: false, stage, reason: 'non_finite_prediction' }
    }
    return {
      ok: true, prediction,
      image: { width: image.cols, height: image.rows }, intrinsics,
    }
  } catch (error) {
    return { ok: false, stage, reason: error instanceof Error ? error.message : String(error) }
  }
}
