import { describe, expect, it } from 'vitest'
import { gazeInZone, NO_ZONE_DWELL, trackZoneDwell, zoneProgress, type ZoneDwell } from './gazeZone'

const VIEWPORT = { width: 1000, height: 800 }
// A strip from x = 900 px to past the right edge, y = 200..600 px.
const STRIP = { left: 900, top: 200, width: Infinity, height: 400 }

describe('gazeInZone', () => {
  it('hits inside the box, including past the screen edge', () => {
    expect(gazeInZone({ x: 0.95, y: 0.5 }, STRIP, VIEWPORT)).toBe(true)
    expect(gazeInZone({ x: 1.2, y: 0.5 }, STRIP, VIEWPORT)).toBe(true) // overshoot still counts
    expect(gazeInZone({ x: 0.95, y: 0.25 }, STRIP, VIEWPORT)).toBe(true) // 200 px: the top edge
  })

  it('misses outside it', () => {
    expect(gazeInZone({ x: 0.85, y: 0.5 }, STRIP, VIEWPORT)).toBe(false)
    expect(gazeInZone({ x: 0.95, y: 0.2 }, STRIP, VIEWPORT)).toBe(false)
    expect(gazeInZone(null, STRIP, VIEWPORT)).toBe(false)
    expect(gazeInZone({ x: NaN, y: 0.5 }, STRIP, VIEWPORT)).toBe(false)
  })
})

describe('zone dwell', () => {
  const feed = (hits: boolean[], maxMisses = 1) => {
    let dwell: ZoneDwell = NO_ZONE_DWELL
    hits.forEach((hit, i) => { dwell = trackZoneDwell(dwell, hit, i * 200, maxMisses) })
    return dwell
  }

  it('counts time from the first sample in the zone', () => {
    const dwell = feed([false, true, true, true])
    expect(dwell).toEqual({ since: 200, misses: 0 })
    expect(zoneProgress(dwell, 1000, 1600)).toBeCloseTo(0.5, 12)
    expect(zoneProgress(dwell, 5000, 1600)).toBe(1)
  })

  it('survives one stray sample but not two in a row', () => {
    expect(feed([true, true, false, true])).toEqual({ since: 0, misses: 0 })
    expect(feed([true, true, false])).toEqual({ since: 0, misses: 1 })
    expect(feed([true, true, false, false])).toEqual(NO_ZONE_DWELL)
    expect(zoneProgress(NO_ZONE_DWELL, 1000, 1600)).toBe(0)
  })
})
