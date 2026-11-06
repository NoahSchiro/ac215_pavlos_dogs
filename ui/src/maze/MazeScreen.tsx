import { useEffect, useRef, useState } from 'react'
import { useCamera } from '../calibration/useCamera'
import { getLandmarker } from '../vision/utils'
import { loadGazeModel, predictGaze } from '../vision/model'
import { captureGazeFrame } from './gazeCapture'
import Maze from './Maze'

type Direction = 'up' | 'down' | 'left' | 'right'

interface Props {
  onBack: () => void
}

const PREDICTION_INTERVAL = 200
const CENTER_MIN = 0.4
const CENTER_MAX = 0.6

function gazeToDirection(x: number, y: number): Direction | null {
  if (!Number.isFinite(x) || !Number.isFinite(y)) return null
  if (x < 0 || x > 1 || y < 0 || y > 1) return null

  const dx = x < CENTER_MIN ? x - CENTER_MIN : x > CENTER_MAX ? x - CENTER_MAX : 0
  const dy = y < CENTER_MIN ? y - CENTER_MIN : y > CENTER_MAX ? y - CENTER_MAX : 0

  if (dx === 0 && dy === 0) return null
  if (Math.abs(dx) > Math.abs(dy)) return dx < 0 ? 'left' : 'right'
  return dy < 0 ? 'up' : 'down'
}

export function MazeScreen({ onBack }: Props) {
  const camera = useCamera()
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)

  const [gazeDirection, setGazeDirection] = useState<Direction | null>(null)
  const [gazeStatus, setGazeStatus] = useState('Starting camera...')

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
    if (status !== 'ready' || !stream) {
      return
    }

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

          const video = videoRef.current
          const canvas = canvasRef.current

          if (video && canvas && video.readyState >= 2) {
            try {
              const image = captureGazeFrame(video, canvas)

              if (image) {
                // Approximate webcam intrinsics; actual focal length varies by device.
                const intrinsics = {
                  focalLengthPx: Math.max(image.cols, image.rows),
                  principalPoint: {
                    x: image.cols / 2,
                    y: image.rows / 2,
                  },
                }

                const result = await predictGaze(
                  landmarker,
                  model,
                  image,
                  intrinsics,
                )

                if (cancelled) return

                if (typeof result === 'string') {
                  setGazeDirection(null)
                  setGazeStatus(`Tracking paused: ${result}`)
                } else {
                  setGazeDirection(gazeToDirection(result.x, result.y))
                  setGazeStatus('Tracking gaze')
                }
              } else {
                setGazeDirection(null)
              }
            } catch (err) {
              if (cancelled) return
              setGazeDirection(null)
              setGazeStatus(
                err instanceof Error ? err.message : 'Gaze prediction failed',
              )
            }
          } else {
            setGazeDirection(null)
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
