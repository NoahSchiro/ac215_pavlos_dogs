import type { ScreenFraction } from '../vision/model'
import type { Viewport } from '../vision/viewport'

/** A run of gaze samples that stay close together: the user is staring somewhere. */
export interface Fixation {
  /** Mean gaze point of the run, as viewport fractions. */
  x: number
  y: number
  samples: number
  /** When the run started, in ms. */
  since: number
}

/**
 * Add a gaze sample to the current fixation, or start a new fixation if the
 * sample is more than `radius` (a fraction of the viewport height) from the
 * fixation's mean. Lost gaze ends the fixation.
 */
export function trackFixation(
  fixation: Fixation | null,
  point: ScreenFraction | null | undefined,
  now: number,
  viewport: Viewport,
  radius: number,
): Fixation | null {
  if (!point || !Number.isFinite(point.x) || !Number.isFinite(point.y) ||
      !(viewport.width > 0) || !(viewport.height > 0)) {
    return null
  }
  if (fixation) {
    // Compare in screen-height units so the radius is round on any screen shape.
    const dx = ((point.x - fixation.x) * viewport.width) / viewport.height
    const dy = point.y - fixation.y
    if (Math.hypot(dx, dy) <= radius) {
      const samples = fixation.samples + 1
      return {
        x: fixation.x + (point.x - fixation.x) / samples,
        y: fixation.y + (point.y - fixation.y) / samples,
        samples,
        since: fixation.since,
      }
    }
  }
  return { x: point.x, y: point.y, samples: 1, since: now }
}

/** True once the fixation has lasted at least `dwellMs`. */
export function isDwelling(fixation: Fixation | null, now: number, dwellMs: number): boolean {
  return fixation !== null && now - fixation.since >= dwellMs
}
