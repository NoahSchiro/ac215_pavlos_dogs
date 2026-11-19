import { useEffect, useRef, useState } from 'react'
import { useCamera } from '../calibration/useCamera'
import type { CalibrationSession } from '../calibration/record'
import { getLandmarker } from '../vision/utils'
import { loadGazeModel, type ScreenFraction } from '../vision/model'
import type { Direction } from '../maze/gazeDirection'
import { predictMazeFrame } from '../maze/predictFrame'
import { GazeKalmanFilter } from './kalman'
import { GAZE_PREDICTION_INTERVAL_MS, GAZE_SMOOTHING } from '../config'

/**
 * Runs the webcam and the gaze model in the browser, and turns each smoothed
 * prediction into a maze direction, using the player's calibration when it
 * applies. Uses the same per-frame step as the solo maze (predictMazeFrame),
 * so race mode steers the same way.
 *
 * Render a hidden <video ref={videoRef}> and <canvas ref={canvasRef}> for it.
 */
export function useGazeTracking(calibration?: CalibrationSession | null) {
  const camera = useCamera()
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)

  const [gazeDirection, setGazeDirection] = useState<Direction | null>(null)
  const [gazeStatus, setGazeStatus] = useState('Starting camera...')
  const [gazePoint, setGazePoint] = useState<ScreenFraction | null>(null)

  const { start, status, stream, error } = camera

  useEffect(() => {
    void start()
  }, [start])

  useEffect(() => {
    const video = videoRef.current
    if (!video || !stream) return

    video.srcObject = stream
    void video.play().catch(() => {})

    return () => {
      video.pause()
      video.srcObject = null
    }
  }, [stream])

  useEffect(() => {
    if (status !== 'ready' || !stream) return

    const controller = new AbortController()
    const { signal } = controller
    let timer: ReturnType<typeof setTimeout> | undefined

    async function run() {
      try {
        setGazeStatus('Loading gaze model...')

        const [landmarker, model] = await Promise.all([
          getLandmarker(),
          loadGazeModel(),
        ])

        if (signal.aborted) return
        const kf = new GazeKalmanFilter(GAZE_SMOOTHING)
        setGazeStatus('Tracking gaze')

        async function predict() {
          if (signal.aborted) return

          const result = await predictMazeFrame({
            video: videoRef.current,
            canvas: canvasRef.current,
            landmarker,
            model,
            stream: stream!,
            calibration,
            filter: kf,
            signal,
          })

          if (signal.aborted) return

          if (result) {
            setGazePoint(result.point)
            setGazeDirection(result.direction)
            if (result.status !== undefined) setGazeStatus(result.status)
          }

          if (!signal.aborted) {
            timer = setTimeout(() => void predict(), GAZE_PREDICTION_INTERVAL_MS)
          }
        }

        void predict()
      } catch (err) {
        if (signal.aborted) return
        setGazeDirection(null)
        setGazeStatus(
          err instanceof Error ? err.message : 'Could not load gaze model',
        )
      }
    }

    void run()

    return () => {
      controller.abort()
      clearTimeout(timer)
      setGazeDirection(null)
      setGazePoint(null)
    }
  }, [status, stream, calibration])

  const statusText = status === 'denied' || status === 'unavailable' ? error : gazeStatus

  return { videoRef, canvasRef, gazeDirection, gazePoint, statusText }
}
