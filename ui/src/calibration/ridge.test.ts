import { describe, expect, it } from 'vitest'
import { applyRidge, fitRidge, type ScreenPoint } from './ridge'

const pts = (xy: number[][]): ScreenPoint[] => xy.map(([x, y]) => ({ x, y }))

const GRID = [0.1, 0.5, 0.9]
const TARGETS = pts(GRID.flatMap((y) => GRID.map((x) => [x, y])))
const PREDICTIONS = pts([
  [0.21, 0.18], [0.47, 0.22], [0.71, 0.27],
  [0.19, 0.46], [0.52, 0.49], [0.76, 0.51],
  [0.24, 0.69], [0.5, 0.74], [0.73, 0.79],
])

describe('fitRidge', () => {
  it('matches the Python fit_ridge it was ported from', () => {
    // Expected values printed by model/src/training/calibration.py::fit_ridge
    // (lamb=0.1) on these exact inputs.
    const fit = fitRidge(PREDICTIONS, TARGETS, 0.1)
    const expected = [1.2237929629820692, -0.13620813620299205, -0.008917361899686739, 1.2356925773779939]
    fit.weight.forEach((w, i) => expect(w).toBeCloseTo(expected[i], 12))
    expect(fit.intercept[0]).toBeCloseTo(-0.08447033393874692, 12)
    expect(fit.intercept[1]).toBeCloseTo(-0.031720164648368576, 12)

    const out = applyRidge(fit, { x: 0.3, y: 0.6 })
    expect(out.x).toBeCloseTo(0.27731713781606177, 12)
    expect(out.y).toBeCloseTo(0.6688329409175301, 12)
  })

  it('recovers a known affine map exactly when unpenalized', () => {
    const map = (p: ScreenPoint) => ({ x: 0.8 * p.x - 0.1 * p.y + 0.05, y: 0.2 * p.x + 0.9 * p.y - 0.1 })
    const fit = fitRidge(TARGETS, TARGETS.map(map), 0)
    const probe = { x: 0.37, y: 0.81 }
    expect(applyRidge(fit, probe).x).toBeCloseTo(map(probe).x, 12)
    expect(applyRidge(fit, probe).y).toBeCloseTo(map(probe).y, 12)
  })

  it('shrinks toward the mean target as lambda grows', () => {
    const fit = fitRidge(PREDICTIONS, TARGETS, 1e12)
    const out = applyRidge(fit, { x: 0.05, y: 0.95 })
    expect(out.x).toBeCloseTo(0.5, 6)
    expect(out.y).toBeCloseTo(0.5, 6)
  })

  it('rejects bad input', () => {
    expect(() => fitRidge(PREDICTIONS, TARGETS, -1)).toThrow('lamb')
    expect(() => fitRidge(PREDICTIONS.slice(1), TARGETS)).toThrow('inputs')
    expect(() => fitRidge([], [])).toThrow('zero points')
    const collinear = pts([[0.1, 0.1], [0.5, 0.5], [0.9, 0.9]])
    expect(() => fitRidge(collinear, collinear, 0)).toThrow('degenerate')
  })
})
