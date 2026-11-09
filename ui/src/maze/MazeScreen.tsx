import { useEffect, useRef, useState } from 'react'
import { useCamera } from '../calibration/useCamera'
import { getLandmarker } from '../vision/utils'
import { loadGazeModel, type ScreenFraction } from '../vision/model'
import { GazeCrosshair } from '../gaze/GazeCrosshair'
import {
  gazeToDirection,
  runGazeInference,
  type Direction,
} from './gazeCapture'
import { captureCameraSettings } from '../calibration/cameraMetadata'
import type { CalibrationSession } from '../calibration/record'
import { prepareGameplayGaze } from './gazeDirection'
import Maze from './Maze'
import { GazeKalmanFilter } from '../gaze/kalman'

interface Props {
  onBack: () => void
  calibration?: CalibrationSession | null
}

const PREDICTION_INTERVAL = 200

export function MazeScreen({ onBack, calibration }: Props) {
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
        const kf = new GazeKalmanFilter()
        setGazeStatus('Tracking gaze')

        async function predict() {
          if (cancelled) return

          try {
            const video = videoRef.current
            const canvas = canvasRef.current

            // Snapshot the actual frame dimensions before inference yields.
            const image = { cols: video?.videoWidth ?? 0, rows: video?.videoHeight ?? 0 }
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
              const prepared = prepareGameplayGaze(result, calibration, captureCameraSettings(stream), image)
              const smoothed = kf.update(prepared?.point ?? null, now)
              // Advance the filter on invalid frames, but do not move or draw stale estimates.
              const gaze = prepared ? smoothed : null
              setGazePoint(gaze)
              setGazeDirection(gaze ? gazeToDirection(gaze.x, gaze.y) : null)
              setGazeStatus(!prepared ? 'Tracking paused: invalid prediction or correction'
                : prepared.calibrated ? 'Tracking gaze with your calibration'
                : calibration ? 'Tracking gaze without calibration. Verify saved camera or recalibrate.'
                : 'Tracking gaze without calibration')
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
            timer = setTimeout(() => void predict(), PREDICTION_INTERVAL)
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
      setGazeDirection(null)
      setGazePoint(null)
    }
  }, [status, stream, calibration])

  return (
    <main className="maze-screen">
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        style={{ display: 'none' }}
      />

      <canvas ref={canvasRef} style={{ display: 'none' }} />

      <Maze gazeDirection={gazeDirection} />

      <GazeCrosshair gaze={gazePoint} />

      <p role="status" aria-live="polite">
        {status === 'denied' || status === 'unavailable'
          ? error
          : gazeStatus}
      </p>

      <button
        type="button"
        className="back-button"
        onClick={onBack}
      >
        Back to Home
      </button>
    </main>
  )
}
