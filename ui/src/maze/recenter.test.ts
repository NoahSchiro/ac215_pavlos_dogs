import { describe, expect, it } from 'vitest'
import {
  addRecenter, correctGaze, correctionAt, holdsPavlos, isRecentering, NO_CORRECTION, recenterProgress,
  sampleRecenter, stareHeld, startRecenter,
  type CorrectionSettings, type GazeCorrection, type RecenterSettings, type RecenterState,
} from './recenter'

const SETTINGS: RecenterSettings = { confirmMs: 1500, captureSamples: 5, holdRadius: 0.08, stareRadius: 0.25, timeoutMs: 10_000 }
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

describe('progress', () => {
  it('fills the first half while confirming and the second while capturing', () => {
    let state = feed(startRecenter(0), [OFF])
    expect(recenterProgress(state, 0, SETTINGS)).toBe(0)
    expect(stareHeld(state, 3)).toBe(false)
    state = feed(state, [OFF, OFF], 200) // stare since 0, now 400 ms
    expect(recenterProgress(state, 750, SETTINGS)).toBeCloseTo(0.25, 12) // 750 of 1500 ms
    expect(stareHeld(state, 3)).toBe(true)
    state = feed(state, repeat(7, OFF), 600) // confirmed at 1600, one sample captured at 1800
    expect(state.phase).toBe('capturing')
    expect(recenterProgress(state, 1800, SETTINGS)).toBeCloseTo(0.6, 12)
    expect(recenterProgress({ phase: 'idle' }, 0, SETTINGS)).toBe(0)
  })
})

const CORRECTION: CorrectionSettings = {
  maxShift: 0.15, falloff: { sigma: 0.3, floor: 0.25, mergeRadius: 0.15, maxAnchors: 4 },
}
const WIDE = { width: 1600, height: 900 }

describe('corrections', () => {
  it('records where the stare was, with the shift', () => {
    // Confirmed at 1600 ms, then five samples captured.
    const state = feed(startRecenter(0), repeat(14, OFF))
    expect(state).toMatchObject({ phase: 'done', at: PAVLOS })
  })

  it('apply in full at the recenter point and fade with distance, down to the floor', () => {
    const correction = addRecenter(NO_CORRECTION, { at: { x: 0.5, y: 0.9 }, shift: { x: 0, y: -0.1 } }, SQUARE, CORRECTION)
    const at = (y: number) => correctionAt(correction, { x: 0.5, y }, SQUARE, CORRECTION).y
    expect(at(0.9)).toBeCloseTo(-0.1, 12)
    // One drive radius away (0.25) it is still mostly there.
    expect(at(0.65)).toBeCloseTo(-0.1 * (0.25 + 0.75 * Math.exp(-(0.25 ** 2) / (2 * 0.3 ** 2))), 12)
    expect(at(0.65)).toBeLessThan(-0.07)
    // The far edge keeps only the floor.
    expect(at(0)).toBeGreaterThan(-0.03)
    expect(at(0)).toBeLessThan(-0.025)
  })

  it('measure distance in screen heights on a wide screen', () => {
    const correction = addRecenter(NO_CORRECTION, { at: { x: 0.5, y: 0.5 }, shift: { x: 0.05, y: 0 } }, WIDE, CORRECTION)
    // 0.3 of the width on a 16:9 screen is 0.53 screen heights: further than 0.3 of the height.
    const across = correctionAt(correction, { x: 0.8, y: 0.5 }, WIDE, CORRECTION).x
    const down = correctionAt(correction, { x: 0.5, y: 0.8 }, WIDE, CORRECTION).x
    expect(across).toBeLessThan(down)
  })

  it('build on the correction already applied, and replace a nearby anchor', () => {
    const at = { x: 0.5, y: 0.9 }
    let correction = addRecenter(NO_CORRECTION, { at, shift: { x: 0, y: -0.06 } }, SQUARE, CORRECTION)
    // Recentering again a little to the side: gaze still reads 0.02 low there.
    const next = { x: 0.52, y: 0.9 }
    const applied = correctionAt(correction, next, SQUARE, CORRECTION).y
    correction = addRecenter(correction, { at: next, shift: { x: 0, y: -0.02 } }, SQUARE, CORRECTION)
    expect(correction).toHaveLength(1)
    expect(correction[0].at).toEqual(next)
    expect(correctionAt(correction, next, SQUARE, CORRECTION).y).toBeCloseTo(applied - 0.02, 12)
  })

  it('keep separate anchors apart, each right at its own point', () => {
    let correction: GazeCorrection = NO_CORRECTION
    correction = addRecenter(correction, { at: { x: 0.5, y: 0.1 }, shift: { x: 0, y: 0.05 } }, SQUARE, CORRECTION)
    const top = correctionAt(correction, { x: 0.5, y: 0.1 }, SQUARE, CORRECTION).y
    const bottomBefore = correctionAt(correction, { x: 0.5, y: 0.9 }, SQUARE, CORRECTION).y
    correction = addRecenter(correction, {
      at: { x: 0.5, y: 0.9 }, shift: { x: 0, y: -0.04 - bottomBefore },
    }, SQUARE, CORRECTION)
    expect(correction).toHaveLength(2)
    expect(correctionAt(correction, { x: 0.5, y: 0.9 }, SQUARE, CORRECTION).y).toBeCloseTo(-0.04, 12)
    // The top moves only by the bottom anchor's faded share.
    const topAfter = correctionAt(correction, { x: 0.5, y: 0.1 }, SQUARE, CORRECTION).y
    expect(Math.abs(topAfter - top)).toBeLessThan(0.02)
  })

  it('keep at most maxAnchors, dropping the oldest', () => {
    let correction: GazeCorrection = NO_CORRECTION
    for (const x of [0.1, 0.4, 0.7, 1.0, 1.3]) {
      correction = addRecenter(correction, { at: { x, y: 0.5 }, shift: { x: 0.01, y: 0 } }, SQUARE, CORRECTION)
    }
    expect(correction.map((anchor) => anchor.at.x)).toEqual([0.4, 0.7, 1.0, 1.3])
  })

  it('cap the total correction at maxShift', () => {
    const correction = addRecenter(NO_CORRECTION, { at: PAVLOS, shift: { x: 0, y: 0.4 } }, SQUARE, CORRECTION)
    expect(correctionAt(correction, PAVLOS, SQUARE, CORRECTION).y).toBeCloseTo(0.15, 12)
    const wide = addRecenter(NO_CORRECTION, { at: PAVLOS, shift: { x: 0.3, y: 0 } }, WIDE, CORRECTION)
    // 0.15 screen heights is 0.15 * 900 / 1600 of the width.
    expect(correctionAt(wide, PAVLOS, WIDE, CORRECTION).x).toBeCloseTo((0.15 * 900) / 1600, 12)
  })

  it('never fold the screen over, even at the largest shift', () => {
    const correction = addRecenter(NO_CORRECTION, { at: PAVLOS, shift: { x: 0, y: 0.15 } }, SQUARE, CORRECTION)
    let previous = -Infinity
    for (let y = -0.3; y <= 1.3; y += 0.005) {
      const corrected = y + correctionAt(correction, { x: 0.5, y }, SQUARE, CORRECTION).y
      expect(corrected).toBeGreaterThan(previous)
      previous = corrected
    }
  })

  it('correct gaze and keep it within the padded screen', () => {
    const correction = addRecenter(NO_CORRECTION, { at: { x: 0.5, y: 0.9 }, shift: { x: 0, y: 0.1 } }, SQUARE, CORRECTION)
    // A look past the bottom edge is kept, up to the padding.
    expect(correctGaze({ x: 0.5, y: 1.0 }, correction, SQUARE, CORRECTION, 0.15)!.y).toBeGreaterThan(1.05)
    expect(correctGaze({ x: 0.5, y: 1.25 }, correction, SQUARE, CORRECTION, 0.15)!.y).toBe(1.15)
    expect(correctGaze({ x: -0.4, y: 0.5 }, NO_CORRECTION, SQUARE, CORRECTION, 0.15)!.x).toBe(-0.15)
    expect(correctGaze(null, correction, SQUARE, CORRECTION, 0.15)).toBeNull()
  })
})
