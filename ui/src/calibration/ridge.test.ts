import { expect, test } from 'vitest'
import { applyCalibration, fitCalibration, type CalibrationFit } from './ridge'

// Existing NumPy reference: model/src/training/calibration.py at 5340538d.
const predictions = [[0.12, 0.22], [0.48, 0.15], [0.83, 0.31], [0.2, 0.64], [0.55, 0.57], [0.91, 0.79], [0.35, 0.88], [0.72, 0.46]].map(([x, y]) => ({ x, y }))
const targets = [[0.18, 0.24], [0.49, 0.12], [0.81, 0.2], [0.3, 0.65], [0.61, 0.52], [0.94, 0.68], [0.48, 0.84], [0.75, 0.36]].map(([x, y]) => ({ x, y }))
const reference = [
  {"lambda": 0.0, "weight": [[0.8730361919563725, -0.1731226519990113], [0.13394354824947893, 0.985475862302382]], "intercept": [0.048714547187323154, 0.04607215823253896], "applied": [-0.025675342850318084, 1.4776751610555772]},
  {"lambda": 0.1, "weight": [[0.7458518952303324, -0.1314498983518295], [0.1252543521250129, 0.8143117207387943]], "intercept": [0.11921670253740829, 0.11041230747170722], "applied": [0.07081722694332665, 1.289883686011568]},
]

test.each(reference)('matches the Python affine fit at lambda $lambda', ({ lambda, weight, intercept, applied }) => {
  const fit = fitCalibration(predictions, targets, lambda)
  fit.weight.flat().forEach((value, i) => expect(value).toBeCloseTo(weight.flat()[i], 12))
  fit.intercept.forEach((value, i) => expect(value).toBeCloseTo(intercept[i], 12))
  const point = applyCalibration({ x: -0.3, y: 1.4 }, fit)
  expect(point.x).toBeCloseTo(applied[0], 12)
  expect(point.y).toBeCloseTo(applied[1], 12)
})

test('defaults to lambda 0.1 and leaves the intercept unpenalized', () => {
  expect(fitCalibration(predictions, targets)).toEqual(fitCalibration(predictions, targets, 0.1))
  const fit = fitCalibration(predictions, targets, 1e12)
  fit.weight.flat().forEach((value) => expect(Math.abs(value)).toBeLessThan(1e-10))
  expect(fit.intercept[0]).toBeCloseTo(0.57, 10)
  expect(fit.intercept[1]).toBeCloseTo(0.45125, 10)
})

test('regularizes constant predictions but rejects a singular unregularized system', () => {
  const raw = [{ x: 0.2, y: 0.3 }, { x: 0.2, y: 0.3 }]
  const expected = [{ x: 0.1, y: 0.2 }, { x: 0.9, y: 0.8 }]
  const corrected = applyCalibration(raw[0], fitCalibration(raw, expected))
  expect(corrected.x).toBeCloseTo(0.5, 12)
  expect(corrected.y).toBeCloseTo(0.5, 12)
  expect(() => fitCalibration(raw, expected, 0)).toThrow(/singular/)
})

test('rejects invalid samples and regularization', () => {
  expect(() => fitCalibration([], [])).toThrow(/at least one/)
  expect(() => fitCalibration(predictions, targets.slice(1))).toThrow(/same length/)
  for (const lambda of [-1, NaN, Infinity]) expect(() => fitCalibration(predictions, targets, lambda)).toThrow(/lambda/)
  expect(() => fitCalibration([{ x: NaN, y: 0 }], [targets[0]])).toThrow(/finite/)
  expect(() => fitCalibration([predictions[0]], [{ x: 0, y: Infinity }])).toThrow(/finite/)
})

test('applies row-vector weights without clipping and rejects invalid corrections', () => {
  const fit: CalibrationFit = { weight: [[1, 2], [3, 4]], intercept: [5, 6], lambda: 0.1 }
  expect(applyCalibration({ x: 7, y: 8 }, fit)).toEqual({ x: 36, y: 52 })
  expect(() => applyCalibration({ x: NaN, y: 0 }, fit)).toThrow(/finite/)
  expect(() => applyCalibration({ x: 0, y: 0 }, { ...fit, intercept: [Infinity, 0] })).toThrow(/finite/)
  expect(() => applyCalibration({ x: 1e308, y: 1e308 }, fit)).toThrow(/non-finite/)
})
