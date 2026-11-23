import { describe, expect, it } from 'vitest'
import {
  addShift, applyShift, holdsPavlos, isRecentering, sampleRecenter, startRecenter,
  type RecenterSettings, type RecenterState,
} from './recenter'

const SETTINGS: RecenterSettings = { confirmMs: 1500, captureSamples: 5, holdRadius: 0.08, maxShift: 0.25, timeoutMs: 10_000 }
const SQUARE = { width: 1000, height: 1000 }
const PAVLOS = { x: 0.5, y: 0.5 }
const STEP_MS = 200 // one prediction

/** Feed gaze points one prediction apart, starting at `t0`. */
function feed(state: RecenterState, points: Array<{ x: number; y: number } | null>, t0 = 0, viewport = SQUARE) {
  points.forEach((point, i) => {
    state = sampleRecenter(state, { point, now: t0 + i * STEP_MS, target: PAVLOS, viewport }, SETTINGS)
  })
  return state
}
const repeat = <T,>(n: number, value: T) => Array.from({ length: n }, () => value)

// Gaze that reads 8% low and 3% right of where the user looks.
const OFF = { x: 0.53, y: 0.58 }

describe('recentering', () => {
  it('does nothing until started', () => {
    expect(feed({ phase: 'idle' }, repeat(20, OFF))).toEqual({ phase: 'idle' })
  })

  it('confirms a steady stare after 1.5 s, then captures 5 predictions', () => {
    let state = feed(startRecenter(0), repeat(8, OFF)) // t = 0 .. 1400 ms
    expect(state.phase).toBe('confirming')
    state = feed(state, [OFF], 1600)
    expect(state).toMatchObject({ phase: 'capturing', samples: [] })
    state = feed(state, repeat(4, OFF), 1800)
    expect(state).toMatchObject({ phase: 'capturing', samples: repeat(4, OFF) })
    state = feed(state, [OFF], 2600)
    expect(state.phase).toBe('done')
    expect(isRecentering(state)).toBe(false)
    expect(holdsPavlos(state)).toBe(true) // until the shifted gaze arrives
    if (state.phase === 'done') {
      expect(state.shift.x).toBeCloseTo(-0.03, 12)
      expect(state.shift.y).toBeCloseTo(-0.08, 12)
    }
  })

  it('takes the median of the captured points, so one stray sample does not skew it', () => {
    const state = feed(startRecenter(0), [
      ...repeat(9, OFF),
      OFF, { x: 0.53, y: 0.64 }, OFF, { x: 0.52, y: 0.57 }, { x: 0.54, y: 0.59 },
    ])
    expect(state.phase).toBe('done')
    if (state.phase === 'done') {
      expect(state.shift.x).toBeCloseTo(-0.03, 12)
      expect(state.shift.y).toBeCloseTo(-0.08, 12)
    }
  })

  it('ignores a stare too far from Pavlos to be at him', () => {
    const state = feed(startRecenter(0), repeat(30, { x: 0.5, y: 0.8 })) // 0.3 of the height away
    expect(state).toMatchObject({ phase: 'confirming', fixation: null })
  })

  it('measures that distance in screen-height units on a wide screen', () => {
    const wide = { width: 2000, height: 1000 }
    // 0.1 of the width = 0.2 of the height: near enough. 0.15 of the width = 0.3: too far.
    expect(feed(startRecenter(0), repeat(9, { x: 0.6, y: 0.5 }), 0, wide).phase).toBe('capturing')
    expect(feed(startRecenter(0), repeat(9, { x: 0.65, y: 0.5 }), 0, wide).phase).toBe('confirming')
  })

  it('restarts the 1.5 s when the stare moves during confirming', () => {
    let state = feed(startRecenter(0), repeat(6, OFF)) // 0 .. 1000 ms
    state = feed(state, repeat(8, PAVLOS), 1200) // a new stare from 1200 ms
    expect(state.phase).toBe('confirming')
    state = feed(state, [PAVLOS], 2800)
    expect(state.phase).toBe('capturing')
  })

  it('goes back to confirming if the stare breaks during capture', () => {
    let state = feed(startRecenter(0), [...repeat(9, OFF), OFF, OFF])
    expect(state.phase).toBe('capturing')
    state = feed(state, [{ x: 0.53, y: 0.4 }], 2200)
    expect(state).toMatchObject({ phase: 'confirming', retry: true })
    state = feed(state, [null], 2400)
    expect(state).toMatchObject({ phase: 'confirming', fixation: null })
  })

  it('gives up after the timeout', () => {
    const state = feed(startRecenter(0), [{ x: 0.9, y: 0.9 }, { x: 0.9, y: 0.9 }], 10_000)
    expect(state).toEqual({ phase: 'gaveUp' })
    expect(isRecentering(state)).toBe(false)
  })
})

describe('shifts', () => {
  it('apply to gaze points and add up across recenters', () => {
    const shift = addShift({ x: 0.01, y: -0.05 }, { x: -0.03, y: -0.02 })
    expect(shift.x).toBeCloseTo(-0.02, 12)
    expect(shift.y).toBeCloseTo(-0.07, 12)
    const point = applyShift({ x: 0.5, y: 0.5 }, shift)
    expect(point.x).toBeCloseTo(0.48, 12)
    expect(point.y).toBeCloseTo(0.43, 12)
    expect(applyShift(null, shift)).toBeNull()
  })
})
