import type { ScreenRect } from './driveAssist'
import type { ScreenFraction } from '../vision/model'
import type { Viewport } from '../vision/viewport'

/** True if a gaze point (viewport fractions) falls inside a box given in CSS pixels. */
export function gazeInZone(point: ScreenFraction | null, zone: ScreenRect, viewport: Viewport): boolean {
  if (!point || !Number.isFinite(point.x) || !Number.isFinite(point.y)) return false
  const x = point.x * viewport.width
  const y = point.y * viewport.height
  return x >= zone.left && x <= zone.left + zone.width && y >= zone.top && y <= zone.top + zone.height
}

/** Time spent looking at a gaze button. */
export interface ZoneDwell {
  /** When the current look started, or null if not looking. */
  since: number | null
  /** Consecutive samples outside the zone during the current look. */
  misses: number
}

export const NO_ZONE_DWELL: ZoneDwell = { since: null, misses: 0 }

/**
 * Add one gaze sample. A look survives up to `maxMisses` stray samples in a
 * row outside the zone (gaze jitter), but more ends it.
 */
export function trackZoneDwell(dwell: ZoneDwell, inZone: boolean, now: number, maxMisses: number): ZoneDwell {
  if (inZone) return { since: dwell.since ?? now, misses: 0 }
  if (dwell.since === null || dwell.misses + 1 > maxMisses) return NO_ZONE_DWELL
  return { since: dwell.since, misses: dwell.misses + 1 }
}

/** How far the look is toward activating the button, from 0 to 1. */
export function zoneProgress(dwell: ZoneDwell, now: number, dwellMs: number): number {
  if (dwell.since === null) return 0
  return Math.min(1, Math.max(0, (now - dwell.since) / dwellMs))
}
