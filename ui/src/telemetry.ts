/**
 * Production telemetry for the gaze loop.
 *
 * Inference runs in the browser, so the server never learns whether the model
 * works: Cloud Run's request logs show page loads, not tracking quality. This
 * aggregates what does matter for each session -- how often a frame yields a
 * gaze, why it does not, how long inference takes, whether the model loaded --
 * and beacons the aggregates to /t on the serving origin. The numbers come from
 * the vision pipeline's own events (vision/events.ts), so no screen has to be
 * instrumented by hand. nginx writes each beacon
 * as one JSON log line; Cloud Logging indexes it, and the log-based metrics in
 * deploy/monitoring/ chart it per model version.
 *
 * Nothing about the user's face or gaze leaves the device: no frames, no
 * landmarks, no gaze coordinates. Only counts, rates and timings, under a
 * random per-page session id.
 */

import { loadRuntimeConfig } from './config'
import { observeInference } from './vision/events'

export type FrameOutcome =
  | 'ok'
  | 'face_not_detected'
  | 'pose_failed'
  | 'reprojection_error'
  | 'eyes_off_face'
  | 'error'

const OUTCOMES: readonly FrameOutcome[] = [
  'ok',
  'face_not_detected',
  'pose_failed',
  'reprojection_error',
  'eyes_off_face',
  'error',
]

/** How often a session reports its window of frames (also sent when the page is hidden). */
export const FLUSH_INTERVAL_MS = 30_000

export type Beacon = (params: URLSearchParams) => void

/** Identity sent with every beacon. */
export interface TelemetryContext {
  sessionId: string
  /** Model version: see modelVersion(). */
  model: string
  appVersion: string
}

/**
 * The model version a URL names: the directory a published model sits in
 * (models/<version>/gazenet.onnx), or 'bundled' for the copy shipped with the UI.
 */
export function modelVersion(modelUrl: string): string {
  const parts = modelUrl.split(/[?#]/)[0].split('/').filter(Boolean)
  const parent = parts.at(-2)
  return parent && parent !== 'models' ? parent : 'bundled'
}

const round = (value: number, digits: number) => Number(value.toFixed(digits))

export class GazeTelemetry {
  private counts = new Map<FrameOutcome, number>()
  private latencySum = 0
  private latencyMax = 0
  private latencyCount = 0
  private windowStart: number
  private readonly send: Beacon
  private readonly context: TelemetryContext
  private readonly now: () => number

  constructor(send: Beacon, context: TelemetryContext, now: () => number = () => performance.now()) {
    this.send = send
    this.context = context
    this.now = now
    this.windowStart = now()
  }

  private beacon(event: string, fields: Record<string, string | number>) {
    const params = new URLSearchParams({
      v: '1',
      ev: event,
      sid: this.context.sessionId,
      model: this.context.model,
      app: this.context.appVersion,
    })
    for (const [key, value] of Object.entries(fields)) params.set(key, String(value))
    this.send(params)
  }

  /** A gaze session started: the model was requested. */
  start(): void {
    this.beacon('start', {})
  }

  /**
   * The model or face landmarker failed to load, so nothing will be tracked.
   * In production this is what a wrong MODEL_URL or a bucket without CORS
   * looks like, and without a report it would be silent.
   */
  recordLoadError(): void {
    this.beacon('load_error', {})
  }

  /** One frame through the pipeline, and how long it took if it ran. */
  recordFrame(outcome: FrameOutcome, latencyMs?: number): void {
    this.counts.set(outcome, (this.counts.get(outcome) ?? 0) + 1)
    if (latencyMs !== undefined && Number.isFinite(latencyMs)) {
      this.latencySum += latencyMs
      this.latencyMax = Math.max(this.latencyMax, latencyMs)
      this.latencyCount++
    }
  }

  /** Report the frames since the last flush, if there were any. */
  flush(): void {
    const frames = [...this.counts.values()].reduce((a, b) => a + b, 0)
    const now = this.now()
    if (frames > 0) {
      const fields: Record<string, number> = {
        frames,
        track_rate: round((this.counts.get('ok') ?? 0) / frames, 3),
        lat_mean: this.latencyCount ? round(this.latencySum / this.latencyCount, 1) : 0,
        lat_max: round(this.latencyMax, 1),
        window_s: round((now - this.windowStart) / 1000, 1),
      }
      for (const outcome of OUTCOMES) fields[outcome] = this.counts.get(outcome) ?? 0
      this.beacon('frames', fields)
    }
    this.counts.clear()
    this.latencySum = this.latencyMax = this.latencyCount = 0
    this.windowStart = now
  }
}

/** Send to /t on the serving origin; survives the page being closed. */
export const sendBeacon: Beacon = (params) => {
  const url = `${import.meta.env.BASE_URL}t?${params}`
  if (!navigator.sendBeacon?.(url)) void fetch(url, { method: 'POST', keepalive: true }).catch(() => {})
}

export function newSessionId(): string {
  return globalThis.crypto?.randomUUID?.() ?? Math.random().toString(36).slice(2)
}

/**
 * Start production telemetry, if this deployment enables it (TELEMETRY in the
 * container's environment). Call once, at startup.
 */
export async function startTelemetry(send: Beacon = sendBeacon): Promise<void> {
  const config = await loadRuntimeConfig()
  if (!config.telemetry) return
  const telemetry = new GazeTelemetry(send, {
    sessionId: newSessionId(),
    model: modelVersion(config.modelUrl),
    appVersion: config.appVersion,
  })
  observeInference((event) => {
    if (event.kind === 'load_start') telemetry.start()
    else if (event.kind === 'load_error') telemetry.recordLoadError()
    else telemetry.recordFrame(event.outcome, event.latencyMs)
  })
  setInterval(() => telemetry.flush(), FLUSH_INTERVAL_MS)
  // Leaving or backgrounding the page may be the last chance to report.
  const flush = () => telemetry.flush()
  window.addEventListener('pagehide', flush)
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'hidden') flush()
  })
}
