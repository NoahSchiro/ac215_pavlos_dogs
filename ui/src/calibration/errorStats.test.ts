import { describe, expect, it } from 'vitest'
import { biasByTarget, errorStats, percentile, residual, suggestDrive } from './errorStats'
import { mazeCellFraction } from '../config'

const WIDE = { width: 1600, height: 800 }
const TOP_LEFT = { x: 0.25, y: 0.25 }
const BOTTOM_LEFT = { x: 0.25, y: 0.75 }

describe('residual', () => {
  it('rescales horizontal error from width to height fractions', () => {
    const r = residual({ x: 0.35, y: 0.3 }, TOP_LEFT, WIDE)
    expect(r.dx).toBeCloseTo(0.2, 12) // 0.1 of 1600 px = 160 px = 0.2 of 800 px
    expect(r.dy).toBeCloseTo(0.05, 12) // positive: below the target
  })
})

describe('percentile', () => {
  it('uses the nearest rank', () => {
    const values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    expect(percentile(values, 0.9)).toBe(11)
    expect(percentile([5], 0.9)).toBe(5)
    expect(percentile([], 0.9)).toBe(0)
  })
})

describe('errorStats', () => {
  it('separates lean from spread on each axis', () => {
    const residuals = [
      { target: TOP_LEFT, dx: 0.02, dy: 0.1 },
      { target: TOP_LEFT, dx: -0.02, dy: 0.06 },
      { target: BOTTOM_LEFT, dx: 0.02, dy: -0.02 },
      { target: BOTTOM_LEFT, dx: -0.02, dy: 0.02 },
    ]
    const stats = errorStats(residuals)
    expect(stats.frames).toBe(4)
    expect(stats.x.bias).toBeCloseTo(0, 12)
    expect(stats.x.meanAbs).toBeCloseTo(0.02, 12)
    expect(stats.y.bias).toBeCloseTo(0.04, 12)
    expect(stats.y.meanAbs).toBeCloseTo(0.05, 12)
    expect(stats.y.p90Abs).toBeCloseTo(0.1, 12)

    const byTarget = biasByTarget(residuals)
    expect(byTarget).toHaveLength(2)
    expect(byTarget[0]).toMatchObject({ target: TOP_LEFT, frames: 2 })
    expect(byTarget[0].dy).toBeCloseTo(0.08, 12) // the top target reads low
    expect(byTarget[1].dy).toBeCloseTo(0, 12)
  })
})

describe('suggestDrive', () => {
  it('adds the p90 miss to two cells of reach, and sizes the cell to cover the miss', () => {
    const stats = {
      frames: 12,
      x: { bias: 0, meanAbs: 0.03, p90Abs: 0.04 },
      y: { bias: 0.05, meanAbs: 0.07, p90Abs: 0.1 },
    }
    const suggestion = suggestDrive(stats, 0.06)
    expect(suggestion.radius).toEqual({ x: 0.16, y: 0.22 })
    expect(suggestion.minCell).toBeCloseTo(0.2, 12)
  })
})

describe('mazeCellFraction', () => {
  it('follows the height-limited maze size', () => {
    expect(mazeCellFraction({ width: 1440, height: 900 })).toBeCloseTo(800 / 15 / 900, 12)
  })

  it('follows the width-limited maze size', () => {
    expect(mazeCellFraction({ width: 500, height: 900 })).toBeCloseTo((0.94 * 500) / 15 / 900, 12)
  })
})
