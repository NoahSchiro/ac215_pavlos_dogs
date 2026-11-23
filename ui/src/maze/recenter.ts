import { trackFixation, type Fixation } from './dwell'
import type { ScreenFraction } from '../vision/model'
import type { Viewport } from '../vision/viewport'

/** An offset added to every gaze point, in viewport fractions. */
export interface GazeShift { x: number; y: number }

export const NO_SHIFT: GazeShift = { x: 0, y: 0 }

export interface RecenterSettings {
  /** How long a steady stare near Pavlos must last before capture starts. */
  confirmMs: number
  /** Gaze samples to capture once confirmed (one per prediction). */
  captureSamples: number
  /** A stare holds while samples stay this close to its mean (fraction of screen height). */
  holdRadius: number
  /** Ignore stares further than this from Pavlos (fraction of screen height): not at him. */
  maxShift: number
  /** Give up if no capture finishes this long after starting. */
  timeoutMs: number
}

export type RecenterState =
  | { phase: 'idle' }
  /**
   * Asked to stare at Pavlos. `fixation` is the current steady stare near him,
   * if any; capture starts once it has lasted `confirmMs`. `retry` is set
   * after a stare broke during capture.
   */
  | { phase: 'confirming'; startedAt: number; fixation: Fixation | null; retry: boolean }
  /** Stare confirmed: collecting `captureSamples` gaze points. */
  | { phase: 'capturing'; startedAt: number; fixation: Fixation; samples: ScreenFraction[] }
  /** Finished: add `shift` to future gaze points. */
  | { phase: 'done'; shift: GazeShift }
  /** No capture within `timeoutMs`. */
  | { phase: 'gaveUp' }

export interface RecenterSample {
  /** The gaze point as the maze sees it (already shifted by any earlier recenter). */
  point: ScreenFraction | null
  now: number
  /** Pavlos's centre, in viewport fractions. */
  target: ScreenFraction
  viewport: Viewport
}

export function startRecenter(now: number): RecenterState {
  return { phase: 'confirming', startedAt: now, fixation: null, retry: false }
}

/** True while recentering is collecting a stare. */
export function isRecentering(state: RecenterState): boolean {
  return state.phase === 'confirming' || state.phase === 'capturing'
}

/**
 * True while Pavlos must hold still: during the stare, and while the result
 * shows, so no move uses a gaze point from before the shift took effect.
 */
export function holdsPavlos(state: RecenterState): boolean {
  return isRecentering(state) || state.phase === 'done'
}

/** Distance between two viewport-fraction points, in screen-height units. */
function heightDistance(a: ScreenFraction, b: ScreenFraction, viewport: Viewport): number {
  return Math.hypot(((a.x - b.x) * viewport.width) / viewport.height, a.y - b.y)
}

function median(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b)
  const mid = Math.floor(sorted.length / 2)
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2
}

/**
 * Advance recentering with one gaze sample. Hands-free: a stare that stays
 * near Pavlos for `confirmMs` starts capture, and the median of the next
 * `captureSamples` points gives the shift that moves the gaze onto him. The
 * stare breaking during capture goes back to confirming.
 */
export function sampleRecenter(state: RecenterState, sample: RecenterSample, settings: RecenterSettings): RecenterState {
  if (state.phase !== 'confirming' && state.phase !== 'capturing') return state
  const { point, now, target, viewport } = sample
  if (now - state.startedAt > settings.timeoutMs) return { phase: 'gaveUp' }

  const near = !!point && Number.isFinite(point.x) && Number.isFinite(point.y) &&
    viewport.width > 0 && viewport.height > 0 &&
    heightDistance(point, target, viewport) <= settings.maxShift
  const previous = state.fixation
  const fixation = near ? trackFixation(previous, point, now, viewport, settings.holdRadius) : null

  if (state.phase === 'confirming') {
    if (fixation && now - fixation.since >= settings.confirmMs) {
      return { phase: 'capturing', startedAt: state.startedAt, fixation, samples: [] }
    }
    return { ...state, fixation }
  }

  // Capturing: a stare that moved away, or started over, breaks the capture.
  if (!fixation || !point || fixation.since !== previous?.since) {
    return { phase: 'confirming', startedAt: state.startedAt, fixation, retry: true }
  }
  const samples = [...state.samples, point]
  if (samples.length < settings.captureSamples) {
    return { ...state, fixation, samples }
  }
  const seen = { x: median(samples.map((p) => p.x)), y: median(samples.map((p) => p.y)) }
  return { phase: 'done', shift: { x: target.x - seen.x, y: target.y - seen.y } }
}

/** Add a shift to a gaze point. */
export function applyShift(point: ScreenFraction, shift: GazeShift): ScreenFraction
export function applyShift(point: ScreenFraction | null, shift: GazeShift): ScreenFraction | null
export function applyShift(point: ScreenFraction | null, shift: GazeShift): ScreenFraction | null {
  return point && { x: point.x + shift.x, y: point.y + shift.y }
}

/** Combine an existing shift with a new correction measured on already-shifted gaze. */
export function addShift(a: GazeShift, b: GazeShift): GazeShift {
  return { x: a.x + b.x, y: a.y + b.y }
}
