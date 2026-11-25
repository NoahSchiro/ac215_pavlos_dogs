import { trackFixation, type Fixation } from './dwell'
import type { ScreenFraction } from '../vision/model'
import type { Viewport } from '../vision/viewport'

/** An offset added to a gaze point, in viewport fractions. */
export interface GazeShift { x: number; y: number }

/**
 * One recenter's correction: `shift` applies in full at `at` (where Pavlos
 * was) and fades with distance from it.
 */
export interface RecenterAnchor { at: ScreenFraction; shift: GazeShift }

/** Every recenter correction since tracking started, oldest first. */
export type GazeCorrection = readonly RecenterAnchor[]

export const NO_CORRECTION: GazeCorrection = []

export interface RecenterSettings {
  /** How long a steady stare near Pavlos must last before capture starts. */
  confirmMs: number
  /** Gaze samples to capture once confirmed (one per prediction). */
  captureSamples: number
  /** A stare holds while samples stay this close to its mean (fraction of screen height). */
  holdRadius: number
  /** Ignore stares further than this from Pavlos (fraction of screen height): not at him. */
  stareRadius: number
  /** Give up if no capture finishes this long after starting. */
  timeoutMs: number
}

export interface CorrectionSettings {
  /** Largest total correction at any point (fraction of screen height). */
  maxShift: number
  falloff: {
    /** Gaussian width of the fade, in screen heights. */
    sigma: number
    /** Share of each shift that still applies far away (0 = purely local). */
    floor: number
    /** A new recenter this close to an older one replaces it (screen heights). */
    mergeRadius: number
    /** Keep at most this many anchors; the oldest goes first. */
    maxAnchors: number
  }
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
  /** Finished: gaze read `shift` away from Pavlos's centre `at` (see addRecenter). */
  | { phase: 'done'; at: ScreenFraction; shift: GazeShift }
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

/**
 * Progress through the stare, from 0 to 1: the first half confirms it, the
 * second half captures.
 */
export function recenterProgress(state: RecenterState, now: number, settings: RecenterSettings): number {
  switch (state.phase) {
    case 'confirming':
      return state.fixation ? Math.min(1, Math.max(0, (now - state.fixation.since) / settings.confirmMs)) / 2 : 0
    case 'capturing':
      return 0.5 + Math.min(1, state.samples.length / settings.captureSamples) / 2
    case 'done':
      return 1
    default:
      return 0
  }
}

/**
 * True once the user has held a stare for `minSamples` predictions. Text near
 * Pavlos should be gone by then, so it is Pavlos they keep looking at.
 */
export function stareHeld(state: RecenterState, minSamples: number): boolean {
  if (state.phase === 'capturing') return true
  return state.phase === 'confirming' && !!state.fixation && state.fixation.samples >= minSamples
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
    heightDistance(point, target, viewport) <= settings.stareRadius
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
  return { phase: 'done', at: target, shift: { x: target.x - seen.x, y: target.y - seen.y } }
}

function falloffWeight(distance: number, falloff: CorrectionSettings['falloff']): number {
  if (!Number.isFinite(distance)) return falloff.floor
  return falloff.floor + (1 - falloff.floor) * Math.exp(-(distance ** 2) / (2 * falloff.sigma ** 2))
}

/** Scale `shift` down so it is no longer than `maxShift` screen heights. */
function capShift(shift: GazeShift, viewport: Viewport, maxShift: number): GazeShift {
  const size = Math.hypot((shift.x * viewport.width) / viewport.height, shift.y)
  if (!(size > maxShift)) return shift
  return { x: (shift.x * maxShift) / size, y: (shift.y * maxShift) / size }
}

/** Weighted sum of the anchors' shifts at `point`, before the cap. */
function sumAt(correction: GazeCorrection, point: ScreenFraction, viewport: Viewport, settings: CorrectionSettings): GazeShift {
  let x = 0
  let y = 0
  for (const anchor of correction) {
    const weight = falloffWeight(heightDistance(point, anchor.at, viewport), settings.falloff)
    x += weight * anchor.shift.x
    y += weight * anchor.shift.y
  }
  return { x, y }
}

/**
 * The correction to add to a gaze point at `point`: each anchor's shift,
 * weighted by a Gaussian fade with a floor, summed and capped at `maxShift`.
 */
export function correctionAt(
  correction: GazeCorrection, point: ScreenFraction, viewport: Viewport, settings: CorrectionSettings,
): GazeShift {
  return capShift(sumAt(correction, point, viewport, settings), viewport, settings.maxShift)
}

/**
 * Correct a gaze point and clamp it to the screen plus `padding` on each side
 * (viewport fractions). The padding leaves room for gaze error at the edges.
 */
export function correctGaze(
  point: ScreenFraction | null, correction: GazeCorrection, viewport: Viewport,
  settings: CorrectionSettings, padding: number,
): ScreenFraction | null {
  if (!point) return null
  const shift = correctionAt(correction, point, viewport, settings)
  const clamp = (v: number) => Math.min(1 + padding, Math.max(0 - padding, v))
  return { x: clamp(point.x + shift.x), y: clamp(point.y + shift.y) }
}

/**
 * Add a finished recenter. `result.shift` was measured on gaze that was
 * already corrected, so the correction at `result.at` becomes what was
 * applied there plus `result.shift`. Older anchors within `mergeRadius` are
 * replaced by the new one.
 */
export function addRecenter(
  correction: GazeCorrection, result: RecenterAnchor, viewport: Viewport, settings: CorrectionSettings,
): GazeCorrection {
  const { at, shift } = result
  const applied = correctionAt(correction, at, viewport, settings)
  const kept = correction.filter((anchor) => !(heightDistance(anchor.at, at, viewport) <= settings.falloff.mergeRadius))
  const others = sumAt(kept, at, viewport, settings)
  // The new anchor has full weight at `at`: make the total there come out right.
  const own = capShift(
    { x: applied.x + shift.x - others.x, y: applied.y + shift.y - others.y }, viewport, settings.maxShift,
  )
  return [...kept, { at, shift: own }].slice(-settings.falloff.maxAnchors)
}
