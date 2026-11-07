import { useEffect, useRef, useState } from 'react'
import type { FaceLandmarker } from '@mediapipe/tasks-vision'
import { correctGaze, fitCalibration, type CalibrationFit } from '../calibration/fit'
import type { CalibrationResult } from '../calibration/points'
import type { RidgeFit } from '../calibration/ridge'
import { useCamera } from '../calibration/useCamera'
import { getLandmarker } from '../vision/utils'
import { loadGazeModel, predictGaze, type GazeModel } from '../vision/model'
import { captureGazeFrame, decodeGazeFrame, estimateIntrinsics } from './gazeCapture'
import Maze from './Maze'

type Direction = 'up' | 'down' | 'left' | 'right'

interface Props {
  onBack: () => void
  /** This session's calibration run, if the user has calibrated. */
  calibration: CalibrationResult | null
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

// One fit per calibration run, kept for the page session so re-entering the
// maze does not re-run the model over every calibration frame. A failed fit is
// evicted rather than cached, so the next visit retries it.
const calibrationFits = new WeakMap<CalibrationResult, Promise<CalibrationFit>>()

function fitFor(
  calibration: CalibrationResult,
  landmarker: FaceLandmarker,
  model: GazeModel,
): Promise<CalibrationFit> {
  let fit = calibrationFits.get(calibration)
  if (!fit) {
    fit = fitCalibration(calibration, async (frame) => {
      const image = await decodeGazeFrame(frame)
      const result = await predictGaze(landmarker, model, image, estimateIntrinsics(image))
      return typeof result === 'string' ? null : result
    })
    calibrationFits.set(calibration, fit)
    fit.catch(() => calibrationFits.delete(calibration))
  }
  return fit
}

export function MazeScreen({ onBack, calibration }: Props) {
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

        let fit: RidgeFit | null = null
        let tracking = 'Tracking gaze (not calibrated)'
        if (calibration) {
          setGazeStatus('Applying calibration...')
          try {
            const result = await fitFor(calibration, landmarker, model)
            if (cancelled) return
            console.info('calibration fit', result)
            if (result.status === 'ok') {
              fit = result.fit
              tracking = 'Tracking gaze (calibrated)'
            } else {
              tracking = `Tracking gaze (calibration unusable: ${
                result.reason === 'flat_predictions'
                  ? 'gaze did not follow the targets'
                  : 'face not found in enough frames'
              }; recalibrate from Home)`
            }
          } catch (err) {
            if (cancelled) return
            console.warn('calibration fit failed', err)
            tracking = 'Tracking gaze (calibration failed; recalibrate from Home)'
          }
        }
        setGazeStatus(tracking)

        async function predict() {
          if (cancelled) return

          const video = videoRef.current
          const canvas = canvasRef.current

          if (video && canvas && video.readyState >= 2) {
            try {
              const image = captureGazeFrame(video, canvas)

              if (image) {
                const result = await predictGaze(
                  landmarker,
                  model,
                  image,
                  estimateIntrinsics(image),
                )

                if (cancelled) return

                if (typeof result === 'string') {
                  setGazeDirection(null)
                  setGazeStatus(`Tracking paused: ${result}`)
                } else {
                  const gaze = correctGaze(fit, result)
                  setGazeDirection(gazeToDirection(gaze.x, gaze.y))
                  setGazeStatus(tracking)
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
