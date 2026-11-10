import { useEffect, useRef, useState } from 'react'
import { useCamera } from '../calibration/useCamera'
import { getLandmarker } from '../vision/utils'
import { loadGazeModel, type ScreenFraction } from '../vision/model'
import { gazeToDirection, runGazeInference, type Direction } from '../maze/gazeCapture'
import { GazeKalmanFilter } from './kalman'
import { GAZE_PREDICTION_INTERVAL_MS, GAZE_SMOOTHING } from '../config'

/**
 * Runs the webcam and the gaze model in the browser, and turns each smoothed
 * prediction into a maze direction. Shared by the solo maze and race mode.
 *
 * Render a hidden <video ref={videoRef}> and <canvas ref={canvasRef}> for it.
 */
export function useGazeTracking() {
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

    let cancelled = false
    let timer: ReturnType<typeof setTimeout> | undefined

    async function run() {
      try {
        setGazeStatus('Loading gaze model...')

        const [landmarker, model] = await Promise.all([
          getLandmarker(),
          loadGazeModel(),
        ])

        if (cancelled) return
        const kf = new GazeKalmanFilter(GAZE_SMOOTHING)
        setGazeStatus('Tracking gaze')

        async function predict() {
          if (cancelled) return

          try {
            const video = videoRef.current
            const canvas = canvasRef.current

            const result =
              video && canvas
                ? await runGazeInference(video, canvas, landmarker, model)
                : null

            if (cancelled) return

            const now = performance.now()

            if (result === null) {
              kf.update(null, now)
              setGazeDirection(null)
              setGazePoint(null)
            } else if (typeof result === 'string') {
              kf.update(null, now)
              setGazeDirection(null)
              setGazePoint(null)
              setGazeStatus(`Tracking paused: ${result}`)
            } else {
              const smoothed = kf.update(result, now)
              setGazeDirection(
                smoothed ? gazeToDirection(smoothed.x, smoothed.y) : null,
              )
              setGazePoint(smoothed)
              setGazeStatus('Tracking gaze')
            }
          } catch (err) {
            if (cancelled) return
            setGazeDirection(null)
            setGazePoint(null)
            setGazeStatus(
              err instanceof Error ? err.message : 'Gaze prediction failed',
            )
          }

          if (!cancelled) {
            timer = setTimeout(() => void predict(), GAZE_PREDICTION_INTERVAL_MS)
          }
        }

        void predict()
      } catch (err) {
        if (cancelled) return
        setGazeDirection(null)
        setGazeStatus(
          err instanceof Error ? err.message : 'Could not load gaze model',
        )
      }
    }

    void run()

    return () => {
      cancelled = true
      clearTimeout(timer)
      setGazePoint(null)
    }
  }, [status, stream])

  const statusText = status === 'denied' || status === 'unavailable' ? error : gazeStatus

  return { videoRef, canvasRef, gazeDirection, gazePoint, statusText }
}
