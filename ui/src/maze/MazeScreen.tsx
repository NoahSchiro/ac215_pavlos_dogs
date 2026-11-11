import { useEffect, useRef, useState } from 'react'
import { useCamera } from '../calibration/useCamera'
import { getLandmarker } from '../vision/utils'
import { loadGazeModel, type ScreenFraction } from '../vision/model'
import { GazeCrosshair } from '../gaze/GazeCrosshair'
import type { Direction } from './gazeDirection'
import type { CalibrationSession } from '../calibration/record'
import { predictMazeFrame } from './predictFrame'
import Maze from './Maze'
import { GazeKalmanFilter } from '../gaze/kalman'
import { enterFullscreen } from '../calibration/fullscreen'
import { currentViewport, sameViewport } from '../vision/viewport'
import { GAZE_PREDICTION_INTERVAL_MS, GAZE_SMOOTHING } from '../config'

interface Props {
  onBack: () => void
  calibration?: CalibrationSession | null
}

const currentLayout = () => ({
  ...currentViewport(),
  fullscreen: !!document.fullscreenElement,
})

export function MazeScreen({ onBack, calibration }: Props) {
  const camera = useCamera()
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const activeRef = useRef(true)

  const [started, setStarted] = useState(false)
  const [starting, setStarting] = useState(false)
  const [gazeDirection, setGazeDirection] = useState<Direction | null>(null)
  const [gazeStatus, setGazeStatus] = useState('Starting camera...')
  const [gazePoint, setGazePoint] = useState<ScreenFraction | null>(null)
  const [layout, setLayout] = useState(currentLayout)

  const initialLayout = useRef<ReturnType<typeof currentLayout> | null>(null)
  const expectedViewport =
    calibration?.cameraVerification === 'verified'
      ? calibration.record.calibration.viewport
      : initialLayout.current

  const fullscreenMode = useRef<boolean | null>(null)

  const layoutMatches =
    sameViewport(expectedViewport, layout) &&
    (fullscreenMode.current === null ||
      layout.fullscreen === fullscreenMode.current)

  const { start, status, stream, error } = camera

  useEffect(() => {
    const update = () => {
      const next = currentLayout()

      // Bind the mode only after reaching the saved size.
      // A failed fullscreen entry may recover later.
      if (
        started &&
        fullscreenMode.current === null &&
        sameViewport(expectedViewport, next)
      ) {
        fullscreenMode.current = next.fullscreen
      }

      setLayout(next)
    }

    window.addEventListener('resize', update)
    document.addEventListener('fullscreenchange', update)

    return () => {
      window.removeEventListener('resize', update)
      document.removeEventListener('fullscreenchange', update)
    }
  }, [calibration, started])

  useEffect(() => {
    activeRef.current = true
    void start()

    return () => {
      activeRef.current = false
    }
  }, [start])

  async function begin() {
    if (status !== 'ready' || !stream || starting) return

    setStarting(true)
    await enterFullscreen()

    if (!activeRef.current) return

    const next = currentLayout()
    initialLayout.current = next

    const expected =
      calibration?.cameraVerification === 'verified'
        ? calibration.record.calibration.viewport
        : next

    fullscreenMode.current =
      sameViewport(expected, next) ? next.fullscreen : null

    setLayout(next)
    setStarted(true)
    setStarting(false)
  }

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
    if (!started) return

    if (!layoutMatches) {
      setGazeStatus(
        'Tracking paused: window size or full screen changed. Restore the calibration window size or recalibrate. Keyboard controls still work.',
      )
      return
    }

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

            if (result.status !== undefined) {
              setGazeStatus(result.status)
            }
          }

          if (!signal.aborted) {
            timer = setTimeout(
              () => void predict(),
              GAZE_PREDICTION_INTERVAL_MS,
            )
          }
        }

        void predict()
      } catch (err) {
        if (signal.aborted) return

        setGazeDirection(null)
        setGazeStatus(
          err instanceof Error
            ? err.message
            : 'Could not load gaze model',
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
  }, [status, stream, calibration, layout, layoutMatches, started])

  return (
    <main className="maze-screen">
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        style={{ display: 'none' }}
      />

      <canvas
        ref={canvasRef}
        style={{ display: 'none' }}
      />

      <Maze gazeDirection={gazeDirection} />

      <GazeCrosshair gaze={gazePoint} />

      <p role="status" aria-live="polite">
        {status === 'denied' || status === 'unavailable'
          ? error
          : !started
            ? starting
              ? 'Entering full screen...'
              : status === 'ready'
                ? 'Camera ready. Click Start game to enter full screen.'
                : 'Starting camera...'
            : gazeStatus}
      </p>

      <div className="maze-side-buttons">
        {!started && (
          <button
            type="button"
            className="back-button"
            disabled={status !== 'ready' || !stream || starting}
            onClick={() => {
              void begin()
            }}
          >
            Start game
          </button>
        )}

        {started && !layout.fullscreen && (
          <p>
            Full screen is inactive or unavailable. Keep the window
            size and position unchanged.
          </p>
        )}

        {started && !layoutMatches && (
          <button
            type="button"
            className="back-button"
            onClick={() => {
              void enterFullscreen()
            }}
          >
            Restore full screen
          </button>
        )}

        <button
          type="button"
          className="back-button"
          onClick={onBack}
        >
          Back to Home
        </button>
      </div>
    </main>
  )
}
