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
import Maze from './Maze'

interface Props {
  onBack: () => void
}

const PREDICTION_INTERVAL = 200

export function MazeScreen({ onBack }: Props) {
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

            if (result === null) {
              setGazeDirection(null)
              setGazePoint(null)
            } else if (typeof result === 'string') {
              setGazeDirection(null)
              setGazePoint(null)
              setGazeStatus(`Tracking paused: ${result}`)
            } else {
              setGazeDirection(gazeToDirection(result.x, result.y))
              setGazePoint(result)
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
      setGazePoint(null)
    }
  }, [status, stream])

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
