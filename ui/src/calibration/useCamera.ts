import { useCallback, useEffect, useRef, useState } from 'react'
import { CAMERA_CONSTRAINTS } from './cameraMetadata'

export type CameraStatus =
  | 'idle'
  | 'requesting'
  | 'ready'
  | 'denied'
  | 'unavailable'

export interface Camera {
  status: CameraStatus
  stream: MediaStream | null
  /** Human readable reason when status is 'denied' or 'unavailable'. */
  error: string | null
  start: () => Promise<void>
  stop: () => void
}

function describeError(err: unknown): { status: CameraStatus; message: string } {
  const name = err instanceof Error ? err.name : ''
  if (name === 'NotAllowedError' || name === 'SecurityError') {
    return {
      status: 'denied',
      message: 'Camera access was denied. Allow camera access in your browser and try again.',
    }
  }
  if (name === 'NotFoundError' || name === 'OverconstrainedError') {
    return { status: 'unavailable', message: 'No camera was found on this device.' }
  }
  if (name === 'NotReadableError') {
    return {
      status: 'unavailable',
      message: 'The camera is in use by another application.',
    }
  }
  return { status: 'unavailable', message: 'The camera could not be started.' }
}

function stopTracks(stream: MediaStream | null | undefined) {
  stream?.getTracks().forEach((t) => t.stop())
}

/**
 * Owns the webcam MediaStream. The stream is stopped on `stop()` and on
 * unmount so the camera indicator never stays on after calibration.
 *
 * Only one getUserMedia request is ever in flight. React Strict Mode mounts
 * components twice in development, and a second overlapping request would
 * open a second camera stream that nothing stops until the browser garbage
 * collects it (Safari keeps the camera on for several seconds). A stream
 * that resolves after `stop()` or unmount is stopped immediately.
 */
export function useCamera(): Camera {
  const [status, setStatus] = useState<CameraStatus>('idle')
  const [stream, setStream] = useState<MediaStream | null>(null)
  const [error, setError] = useState<string | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const pendingRef = useRef(false)
  const mountedRef = useRef(true)
  // Incremented on stop() so a request started before it is discarded.
  const generationRef = useRef(0)

  const stop = useCallback(() => {
    generationRef.current++
    stopTracks(streamRef.current)
    streamRef.current = null
    setStream(null)
    setStatus('idle')
    setError(null)
  }, [])

  const start = useCallback(async () => {
    if (streamRef.current || pendingRef.current) return
    setStatus('requesting')
    setError(null)

    const media = navigator.mediaDevices
    if (!media?.getUserMedia) {
      setStatus('unavailable')
      setError('This browser does not support camera access.')
      return
    }

    pendingRef.current = true
    const generation = generationRef.current
    try {
      const s = await media.getUserMedia(CAMERA_CONSTRAINTS)
      const stale = !mountedRef.current || generation !== generationRef.current
      if (stale || streamRef.current) {
        stopTracks(s)
        return
      }
      streamRef.current = s
      setStream(s)
      setStatus('ready')
    } catch (err) {
      if (!mountedRef.current || generation !== generationRef.current) return
      const { status: next, message } = describeError(err)
      setStatus(next)
      setError(message)
    } finally {
      pendingRef.current = false
    }
  }, [])

  useEffect(() => {
    mountedRef.current = true
    return () => {
      mountedRef.current = false
      stopTracks(streamRef.current)
      streamRef.current = null
    }
  }, [])

  return { status, stream, error, start, stop }
}
