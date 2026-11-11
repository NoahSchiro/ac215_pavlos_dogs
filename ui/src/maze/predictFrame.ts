import type { FaceLandmarker } from '@mediapipe/tasks-vision'
import { captureCameraSettings } from '../vision/camera'
import type { CalibrationSession } from '../calibration/record'
import type { GazeKalmanFilter } from '../gaze/kalman'
import type { GazeModel, ScreenFraction } from '../vision/model'
import { runGazeInference } from './gazeCapture'
import { gazeToDirection, prepareGameplayGaze, type Direction } from './gazeDirection'

interface MazeFrameInput {
  video: HTMLVideoElement | null
  canvas: HTMLCanvasElement | null
  landmarker: FaceLandmarker
  model: GazeModel
  stream: MediaStream
  calibration?: CalibrationSession | null
  filter: GazeKalmanFilter
  signal: AbortSignal
}

interface MazeFramePrediction {
  point: ScreenFraction | null
  direction: Direction | null
  status?: string
}

/** Predict one gameplay frame. Aborted work must not advance the filter. */
export async function predictMazeFrame({
  video, canvas, landmarker, model, stream, calibration, filter, signal,
}: MazeFrameInput): Promise<MazeFramePrediction | null> {
  if (signal.aborted) return null
  try {
    const image = { cols: video?.videoWidth ?? 0, rows: video?.videoHeight ?? 0 }
    const raw = video && canvas ? await runGazeInference(video, canvas, landmarker, model) : null
    if (signal.aborted) return null
    const now = performance.now()

    if (raw === null || typeof raw === 'string') {
      filter.update(null, now)
      return { point: null, direction: null,
        ...(raw === null ? {} : { status: `Tracking paused: ${raw}` }) }
    }

    const prepared = prepareGameplayGaze(raw, calibration, captureCameraSettings(stream), image)
    const smoothed = filter.update(prepared?.point ?? null, now)
    // Advance the filter on invalid frames without displaying or using stale estimates.
    const point = prepared ? smoothed : null
    return {
      point,
      direction: point ? gazeToDirection(point.x, point.y) : null,
      status: !prepared ? 'Tracking paused: invalid prediction or correction'
        : prepared.calibrated ? 'Tracking gaze with your calibration'
        : calibration ? 'Tracking gaze without calibration. Verify saved camera or recalibrate.'
        : 'Tracking gaze without calibration',
    }
  } catch (error) {
    if (signal.aborted) return null
    return { point: null, direction: null,
      status: error instanceof Error ? error.message : 'Gaze prediction failed' }
  }
}
