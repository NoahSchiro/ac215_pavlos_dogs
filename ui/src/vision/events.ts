/**
 * What the vision pipeline reports about itself, for production monitoring.
 *
 * The pipeline emits these and ../telemetry.ts listens. They live here,
 * rather than in the screens that call the pipeline, so every caller is
 * measured the same way and monitoring needs no UI changes.
 */
import type { DropReason } from './utils'

export type InferenceEvent =
  /** The model was requested: a gaze session is starting. */
  | { kind: 'load_start' }
  /** The model or the face landmarker failed to load; nothing will track. */
  | { kind: 'load_error' }
  /** One webcam frame through predictGaze, and how long it took. */
  | { kind: 'frame'; outcome: 'ok' | DropReason | 'error'; latencyMs: number }

let observer: ((event: InferenceEvent) => void) | null = null

/** Receive InferenceEvents (one observer at a time); null stops. */
export function observeInference(listener: ((event: InferenceEvent) => void) | null): void {
  observer = listener
}

export function reportInference(event: InferenceEvent): void {
  try {
    observer?.(event)
  } catch {
    // Monitoring must never break inference.
  }
}
