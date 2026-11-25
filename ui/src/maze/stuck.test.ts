import { describe, expect, it } from 'vitest'
import { isStuckPrompt, NEW_STUCK, quietStuck, trackStuck, type StuckSettings, type StuckState } from './stuck'

const SETTINGS: StuckSettings = { stillMs: 4000, windowMs: 4000, rockMoves: 4, rockCells: 2, promptMs: 6000, cooldownMs: 15_000 }
const A = { row: 4, col: 4 }
const B = { row: 4, col: 5 }

/** One sample every 200 ms from `t0`; `cells[i]` is Pavlos's cell, `trying` applies throughout. */
function feed(state: StuckState, cells: Array<{ row: number; col: number }>, trying: boolean, t0 = 0) {
  cells.forEach((player, i) => { state = trackStuck(state, { now: t0 + i * 200, player, trying }, SETTINGS) })
  return state
}
const stay = (n: number, cell = A) => Array.from({ length: n }, () => cell)

describe('stuck detection', () => {
  it('prompts when Pavlos stays put for 4 s while the user steers near him', () => {
    expect(isStuckPrompt(feed(NEW_STUCK, stay(20), true))).toBe(false) // 0 .. 3800 ms
    expect(isStuckPrompt(feed(NEW_STUCK, stay(21), true))).toBe(true) // 4000 ms
  })

  it('does not prompt while the user looks at Pavlos (stopping) or far away', () => {
    expect(isStuckPrompt(feed(NEW_STUCK, stay(60), false))).toBe(false)
  })

  it('restarts the clock when the user stops trying for a moment, or Pavlos moves', () => {
    let state = feed(NEW_STUCK, stay(15), true) // 0 .. 2800
    state = feed(state, [A], false, 3000)
    state = feed(state, stay(16), true, 3200) // 3200 .. 6200: 3 s of trying
    expect(isStuckPrompt(state)).toBe(false)

    state = feed(NEW_STUCK, [...stay(15), ...stay(16, { row: 5, col: 4 })], true) // moved at 3000
    expect(isStuckPrompt(state)).toBe(false)
  })

  it('prompts when Pavlos rocks between two cells', () => {
    expect(isStuckPrompt(feed(NEW_STUCK, [A, B, A, B, A], false))).toBe(true) // 4 moves in 1 s
    expect(isStuckPrompt(feed(NEW_STUCK, [A, B, A, B], false))).toBe(false) // only 3 moves
  })

  it('does not count steady progress as rocking', () => {
    const path = [A, B, { row: 4, col: 6 }, { row: 4, col: 7 }, { row: 4, col: 8 }]
    expect(isStuckPrompt(feed(NEW_STUCK, path, false))).toBe(false)
  })

  it('ignores moves older than the window', () => {
    const slow = [A, ...stay(10, B), ...stay(10, A), ...stay(10, B), ...stay(10, A)] // a move every 2 s
    expect(isStuckPrompt(feed(NEW_STUCK, slow, false))).toBe(false)
  })

  it('shows the prompt for 6 s, then stays quiet for 15 s', () => {
    let state = feed(NEW_STUCK, stay(21), true) // prompt at 4000
    state = feed(state, stay(29), true, 4200) // .. 9800
    expect(isStuckPrompt(state)).toBe(true)
    state = feed(state, [A], true, 10_000)
    expect(isStuckPrompt(state)).toBe(false)
    expect(state.quietUntil).toBe(25_000)
    state = feed(state, stay(74), true, 10_200) // .. 24_800: still stuck, but quiet
    expect(isStuckPrompt(state)).toBe(false)
    state = feed(state, [A], true, 25_000)
    expect(isStuckPrompt(state)).toBe(true)
  })

  it('goes quiet when recentering starts', () => {
    let state = feed(NEW_STUCK, stay(21), true)
    state = quietStuck(state, 4200, SETTINGS)
    expect(isStuckPrompt(state)).toBe(false)
    expect(isStuckPrompt(feed(state, stay(20), true, 4400))).toBe(false)
  })
})
