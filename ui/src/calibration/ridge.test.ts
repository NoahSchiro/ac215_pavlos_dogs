import { describe, expect, test } from 'vitest'
import reference from './fixtures/ridge-reference.json'
import { applyCalibration, fitCalibration, type CalibrationFit } from './ridge'
import type { NormalizedPoint } from './points'

const tolerance = reference.absoluteTolerance
const points = (rows: readonly (readonly number[])[]): NormalizedPoint[] =>
  rows.map(([x, y]) => ({ x, y }))
const predictions = points([[0, 0], [1, 0], [0, 1], [1, 1]])
const targets = points([[0.1, 0.2], [0.8, 0.1], [0.2, 0.9], [0.9, 0.8]])

function expectNear(actual: number, expected: number) {
  expect(Math.abs(actual - expected)).toBeLessThanOrEqual(tolerance)
}

describe('Python reference parity (absolute tolerance 1e-12)', () => {
  test.each(reference.cases)('$name', (fixture) => {
    const fit = fitCalibration(points(fixture.predictions), points(fixture.targets), fixture.lambda)
    expect(fit.lambda).toBe(fixture.lambda)
    for (let i = 0; i < 2; i++) {
      for (let j = 0; j < 2; j++) expectNear(fit.weight[i][j], fixture.weight[i][j])
      expectNear(fit.intercept[i], fixture.intercept[i])
    }
    reference.probes.forEach(([x, y], i) => {
      const actual = applyCalibration({ x, y }, fit)
      expectNear(actual.x, fixture.applied[i][0])
      expectNear(actual.y, fixture.applied[i][1])
    })
  })
})

test('recovers an asymmetric affine mapping, including cross-axis weights and intercept', () => {
  const grid = [0.1, 0.5, 0.9].flatMap((x) => [0.1, 0.5, 0.9].map((y) => ({ x, y })))
  const mapped = grid.map(({ x, y }) => ({ x: 0.8 * x + 0.1 * y + 0.05, y: -0.2 * x + 0.9 * y - 0.1 }))
  const fit = fitCalibration(grid, mapped, 0)
  ;[0.8, -0.2, 0.1, 0.9].forEach((value, i) => expectNear(fit.weight.flat()[i], value))
  expectNear(fit.intercept[0], 0.05)
  expectNear(fit.intercept[1], -0.1)
  const heldOut = applyCalibration({ x: 0.33, y: 0.71 }, fit)
  expectNear(heldOut.x, 0.385)
  expectNear(heldOut.y, 0.473)
})

test('defaults to lambda 0.1', () => {
  expect(fitCalibration(predictions, targets)).toEqual(fitCalibration(predictions, targets, 0.1))
})

test('large lambda shrinks weights toward zero without penalizing the target mean', () => {
  const fit = fitCalibration(predictions, targets, 1e12)
  fit.weight.flat().forEach((value) => expect(Math.abs(value)).toBeLessThan(1e-10))
  const correctedMean = applyCalibration({ x: 0.5, y: 0.5 }, fit)
  expectNear(correctedMean.x, 0.5)
  expectNear(correctedMean.y, 0.5)
})

test('translating targets changes only the unpenalized intercept', () => {
  const fit = fitCalibration(predictions, targets)
  const translated = fitCalibration(predictions, targets.map(({ x, y }) => ({ x: x + 10, y: y - 4 })))
  fit.weight.flat().forEach((value, i) => expectNear(translated.weight.flat()[i], value))
  expectNear(translated.intercept[0], fit.intercept[0] + 10)
  expectNear(translated.intercept[1], fit.intercept[1] - 4)
})

test('constant targets produce a mean-only correction even without regularization', () => {
  const fit = fitCalibration(predictions, predictions.map(() => ({ x: 0.4, y: 0.6 })), 0)
  expect(fit.weight.flat()).toEqual([0, 0, 0, 0])
  expect(applyCalibration({ x: -2, y: 3 }, fit)).toEqual({ x: 0.4, y: 0.6 })
})

test('fitting and application accept frozen inputs without mutating them', () => {
  const raw = Object.freeze(predictions.map((point) => Object.freeze({ ...point })))
  const expected = Object.freeze(targets.map((point) => Object.freeze({ ...point })))
  const fit = fitCalibration(raw, expected)
  Object.freeze(fit.weight[0])
  Object.freeze(fit.weight[1])
  Object.freeze(fit.weight)
  Object.freeze(fit.intercept)
  Object.freeze(fit)
  const before = JSON.stringify({ raw, expected, fit })
  applyCalibration(raw[0], fit)
  expect(JSON.stringify({ raw, expected, fit })).toBe(before)
})

describe('invalid fits', () => {
  test('rejects empty samples', () => {
    expect(() => fitCalibration([], [])).toThrow(/at least one/)
  })

  test.each([[predictions, targets.slice(1)], [predictions.slice(1), targets]])(
    'rejects mismatched sample lengths', (raw, expected) => {
      expect(() => fitCalibration(raw, expected)).toThrow(/same length/)
    },
  )

  test.each([-1, NaN, Infinity, -Infinity])('rejects lambda %s', (lambda) => {
    expect(() => fitCalibration(predictions, targets, lambda)).toThrow(/lambda.*finite.*non-negative/)
  })

  for (const source of ['predictions', 'targets'] as const) {
    for (const axis of ['x', 'y'] as const) {
      test.each([NaN, Infinity, -Infinity])(`rejects non-finite ${source}.${axis}: %s`, (value) => {
        const invalid = [...(source === 'predictions' ? predictions : targets)]
        invalid[1] = { ...invalid[1], [axis]: value }
        expect(() => fitCalibration(
          source === 'predictions' ? invalid : predictions,
          source === 'targets' ? invalid : targets,
        )).toThrow(`${source}[1]`)
      })
    }
  }

  test.each([
    [{ x: 0.2, y: 0.3 }],
    [{ x: 0.2, y: 0.3 }, { x: 0.2, y: 0.3 }, { x: 0.2, y: 0.3 }],
    [{ x: 0, y: 0 }, { x: 0.5, y: 1 }, { x: 1, y: 2 }],
    [{ x: 0, y: 0 }, { x: 0.5, y: 1 }, { x: 1, y: 2 + 1e-12 }],
  ].map((raw) => ({ raw })))('rejects a singular or numerically singular unregularized system: $raw', ({ raw }) => {
    expect(() => fitCalibration(raw, raw, 0)).toThrow(/singular/)
  })

  test('rejects arithmetic overflow from finite inputs', () => {
    const huge = [{ x: 1e308, y: 0 }, { x: -1e308, y: 1 }, { x: 0, y: 0 }]
    expect(() => fitCalibration(huge, predictions.slice(0, 3))).toThrow(/non-finite/)
  })

  test('reports a missing coordinate as an invalid point', () => {
    const malformed = [{ x: 0.5 }] as NormalizedPoint[]
    expect(() => fitCalibration(malformed, [targets[0]])).toThrow(/predictions\[0\].*finite/)
  })
})

describe('applying a fit', () => {
  const fit: CalibrationFit = { weight: [[1, 2], [3, 4]], intercept: [5, 6], lambda: 0.1 }

  test('uses row vectors and leaves finite coordinates unclipped', () => {
    expect(applyCalibration({ x: 7, y: 8 }, fit)).toEqual({ x: 36, y: 52 })
  })

  test.each([{ x: NaN, y: 0 }, { x: 0, y: Infinity }])('rejects a non-finite point: %j', (point) => {
    expect(() => applyCalibration(point, fit)).toThrow(/point.*finite/)
  })

  test.each([
    { ...fit, weight: [[NaN, 0], [0, 1]] },
    { ...fit, weight: [[1, 0], [0, Infinity]] },
    { ...fit, intercept: [0, NaN] },
    { ...fit, intercept: [Infinity, 0] },
  ])('rejects non-finite fit coefficients: %j', (invalid) => {
    expect(() => applyCalibration({ x: 0.5, y: 0.5 }, invalid as unknown as CalibrationFit)).toThrow(/fit.*finite/)
  })

  test.each([
    { ...fit, weight: [[1, 0]] },
    { ...fit, weight: [[1, 0], [0, 1, 2]] },
    { ...fit, intercept: [0] },
  ])('rejects malformed fit dimensions: %j', (invalid) => {
    expect(() => applyCalibration({ x: 0.5, y: 0.5 }, invalid as unknown as CalibrationFit)).toThrow(/fit.*2/)
  })

  test('rejects a non-finite correction produced by finite inputs', () => {
    expect(() => applyCalibration({ x: 1e308, y: 1e308 }, fit)).toThrow(/non-finite/)
  })

  test('rejects missing matrix rows with a clear shape error', () => {
    const weight = [[1, 0]]
    weight.length = 2
    expect(() => applyCalibration({ x: 0.5, y: 0.5 }, {
      ...fit, weight,
    } as unknown as CalibrationFit)).toThrow(/fit.*2x2/)
  })
})
