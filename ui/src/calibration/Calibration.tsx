import { useCallback, useEffect, useRef, useState } from 'react'
import { captureBurst } from './capture'
import { captureCameraSettings } from './cameraMetadata'
import { Crosshair } from './Crosshair'
import { enterFullscreen } from './fullscreen'
import {
  CALIBRATION_POINTS,
  VALIDATION_POINTS,
  FRAMES_PER_POINT,
  FRAME_INTERVAL_MS,
  type CalibrationResult,
  type CalibrationSample,
} from './points'
import { useCamera } from './useCamera'
import { currentViewport, sameViewport, type Viewport } from './viewport'
import './calibration.css'

interface Props {
  onComplete: (result: CalibrationResult) => void
  onCancel: () => void
  mode?: 'calibration' | 'validation'
  expectedViewport?: Viewport | null
}

/**
 * Full-viewport calibration screen.
 *
 * The camera is opened first, while the page is still windowed, so the
 * browser's permission prompt stays visible (Safari hides it in full screen).
 * Once the camera is ready the user clicks Begin, which enters full screen
 * and shows the first crosshair. Space captures a burst of frames labeled
 * with the current target, Escape cancels.
 */
export function Calibration({ onComplete, onCancel, mode = 'calibration', expectedViewport }: Props) {
  const camera = useCamera()
  const videoRef = useRef<HTMLVideoElement>(null)
  const samplesRef = useRef<CalibrationSample[]>([])
  const capturingRef = useRef(false)
  const doneRef = useRef(false)
  const activeRef = useRef(true)
  const beginningRef = useRef(false)
  const layoutRef = useRef<(Viewport & { fullscreen: boolean }) | null>(null)

  const [started, setStarted] = useState(false)
  const [pointIndex, setPointIndex] = useState(0)
  const [capturing, setCapturing] = useState(false)
  const [captureError, setCaptureError] = useState<string | null>(null)
  const [layoutError, setLayoutError] = useState<string | null>(null)

  const { start, stop, status, stream } = camera
  const targets = mode === 'validation' ? VALIDATION_POINTS : CALIBRATION_POINTS
  const total = targets.length
  const point = targets[pointIndex]

  useEffect(() => {
    activeRef.current = true
    return () => {
      activeRef.current = false
    }
  }, [])

  useEffect(() => {
    const checkLayout = () => {
      const layout = layoutRef.current
      if (layout && (!sameViewport(layout, currentViewport()) || layout.fullscreen !== !!document.fullscreenElement)) {
        layoutRef.current = null
        setLayoutError('Window size or full screen changed. Go back and restart calibration.')
      }
    }
    window.addEventListener('resize', checkLayout)
    document.addEventListener('fullscreenchange', checkLayout)
    return () => {
      window.removeEventListener('resize', checkLayout)
      document.removeEventListener('fullscreenchange', checkLayout)
    }
  }, [])

  // Open the camera as soon as the screen mounts.
  useEffect(() => {
    void start()
  }, [start])

  // Attach the stream to the hidden video element.
  useEffect(() => {
    const video = videoRef.current
    if (!video || !stream) return
    video.srcObject = stream
    void video.play?.()?.catch(() => {})
    return () => {
      // Pause and detach so Safari releases the capture pipeline promptly.
      video.pause?.()
      video.srcObject = null
    }
  }, [stream])

  const cancel = useCallback(() => {
    if (doneRef.current) return
    doneRef.current = true
    stop()
    onCancel()
  }, [stop, onCancel])

  const begin = useCallback(async () => {
    if (status !== 'ready' || beginningRef.current || doneRef.current) return
    beginningRef.current = true
    await enterFullscreen()
    beginningRef.current = false
    if (!activeRef.current || doneRef.current) return
    const viewport = currentViewport()
    if (expectedViewport !== undefined && !sameViewport(expectedViewport, viewport)) {
      setLayoutError('Window size does not match calibration. Go back and restore the original size or recalibrate.')
      return
    }
    layoutRef.current = { ...viewport, fullscreen: !!document.fullscreenElement }
    setStarted(true)
  }, [status, expectedViewport])

  const captureCurrent = useCallback(async () => {
    const video = videoRef.current
    const layout = layoutRef.current
    if (!video || !layout || !started || capturingRef.current || doneRef.current || status !== 'ready') {
      return
    }

    capturingRef.current = true
    setCapturing(true)
    setCaptureError(null)

    const index = pointIndex
    const target = targets[index]
    try {
      const frames = await captureBurst(video, FRAMES_PER_POINT, FRAME_INTERVAL_MS)
      // Camera shutdown cannot cancel JPEG encoding already in progress.
      if (!activeRef.current || doneRef.current) return
      if (layoutRef.current !== layout || !sameViewport(layout, currentViewport()) ||
          layout.fullscreen !== !!document.fullscreenElement) {
        layoutRef.current = null
        setLayoutError('Window size or full screen changed. Go back and restart calibration.')
        return
      }
      const now = Date.now()
      const batch: CalibrationSample[] = frames.map((image) => ({
        image,
        target,
        pointIndex: index,
        viewportWidth: layout.width,
        viewportHeight: layout.height,
        timestamp: now,
      }))
      samplesRef.current = [...samplesRef.current, ...batch]

      if (index + 1 >= total) {
        doneRef.current = true
        const capturedCamera = captureCameraSettings(stream)
        stop()
        onComplete({ samples: samplesRef.current, completedAt: now, camera: capturedCamera })
      } else {
        setPointIndex(index + 1)
      }
    } catch {
      if (activeRef.current && !doneRef.current) {
        setCaptureError('Capture failed. Make sure your face is visible and press Space again.')
      }
    } finally {
      capturingRef.current = false
      if (activeRef.current && !doneRef.current) setCapturing(false)
    }
  }, [pointIndex, started, status, stop, stream, onComplete, total, targets])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.code === 'Space' || e.key === ' ') {
        if (e.target instanceof Element && e.target.closest('button')) return
        e.preventDefault()
        if (e.repeat) return
        if (started) void captureCurrent()
        else void begin()
      } else if (e.key === 'Escape') {
        e.preventDefault()
        cancel()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [captureCurrent, cancel, begin, started])

  const cameraFailed = status === 'denied' || status === 'unavailable'

  return (
    <div className="calibration" role="application" aria-label={mode === 'validation' ? 'Gaze validation' : 'Gaze calibration'}>
      <video ref={videoRef} autoPlay playsInline muted />

      {cameraFailed && (
        <div className="calibration-status" role="alert">
          <h2>Camera unavailable</h2>
          <p>{camera.error}</p>
          <button type="button" onClick={() => void start()}>
            Retry
          </button>
          <button type="button" onClick={cancel}>
            Back
          </button>
        </div>
      )}

      {status === 'requesting' && (
        <div className="calibration-status">
          <h2>Starting camera</h2>
          <p>Allow camera access when your browser asks.</p>
        </div>
      )}

      {layoutError && <div className="calibration-status" role="alert">
        <p>{layoutError}</p>
        <button type="button" onClick={cancel}>Back</button>
      </div>}

      {status === 'ready' && !started && !layoutError && (
        <div className="calibration-status">
          <h2>Camera ready</h2>
          <p>
            You will see {total} targets, one at a time. Look at each one and press Space.
            The page will go full screen.
          </p>
          {mode === 'validation' && <p>These new frames only check accuracy. They do not change your calibration.</p>}
          <button type="button" onClick={begin} autoFocus>
            Begin
          </button>
          <button type="button" onClick={cancel}>
            Back
          </button>
        </div>
      )}

      {status === 'ready' && started && !layoutError && (
        <>
          <Crosshair point={point} />
          <div className="calibration-hud">
            {!layoutRef.current?.fullscreen && <p>Full screen is unavailable. Keep this window size and position unchanged.</p>}
            <div>
              <strong>
                Point {pointIndex + 1} of {total}
              </strong>
            </div>
            {capturing ? (
              <div className="capturing">Capturing, hold still</div>
            ) : (
              <div>Look at the target and press Space. Press Escape to cancel.</div>
            )}
            {captureError && <div className="capture-error">{captureError}</div>}
          </div>
        </>
      )}
    </div>
  )
}
