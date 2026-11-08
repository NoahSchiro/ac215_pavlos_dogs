import type { NormalizedPoint } from './points'

/** Row-vector affine map: corrected = raw @ weight + intercept. */
export interface CalibrationFit {
  /** Rows are input axes; columns are corrected output axes. */
  readonly weight: readonly [readonly [number, number], readonly [number, number]]
  readonly intercept: readonly [number, number]
  readonly lambda: number
}

function validatePoint(point: NormalizedPoint, label: string): void {
  if (!point || !Number.isFinite(point.x) || !Number.isFinite(point.y)) {
    throw new Error(`${label} must have finite x and y coordinates`)
  }
}

function validateFit(fit: CalibrationFit): void {
  if (
    !fit || !Array.isArray(fit.weight) || fit.weight.length !== 2 ||
    !Array.isArray(fit.weight[0]) || fit.weight[0].length !== 2 ||
    !Array.isArray(fit.weight[1]) || fit.weight[1].length !== 2 ||
    !Array.isArray(fit.intercept) || fit.intercept.length !== 2
  ) {
    throw new Error('Calibration fit must have a 2x2 weight and a 2-element intercept')
  }
  const coefficients = [
    fit.weight[0][0], fit.weight[0][1], fit.weight[1][0], fit.weight[1][1],
    fit.intercept[0], fit.intercept[1],
  ]
  if (!coefficients.every(Number.isFinite)) {
    throw new Error('Calibration fit coefficients must be finite')
  }
}

/**
 * Minimize sum(||target - raw @ W - b||²) + lambda * ||W||².
 * Center both inputs to leave b unpenalized. Lambda is not divided by N.
 * Positive lambda supports even a single sample or constant predictions.
 * Throw on invalid input, overflow, or a numerically singular system.
 * No clipping, target coverage, or image/runtime logic belongs here.
 *
 * Reference: PR #60, model/src/training/calibration.py, commit 5340538.
 */
export function fitCalibration(
  predictions: readonly NormalizedPoint[],
  targets: readonly NormalizedPoint[],
  lambda = 0.1,
): CalibrationFit {
  if (predictions.length !== targets.length) {
    throw new Error('Predictions and targets must have the same length')
  }
  const count = predictions.length
  if (count === 0) throw new Error('Calibration requires at least one sample pair')
  if (!Number.isFinite(lambda) || lambda < 0) {
    throw new Error('lambda must be finite and non-negative')
  }

  let meanX = 0, meanY = 0, meanTargetX = 0, meanTargetY = 0
  for (let i = 0; i < count; i++) {
    validatePoint(predictions[i], `predictions[${i}]`)
    validatePoint(targets[i], `targets[${i}]`)
    meanX += predictions[i].x / count
    meanY += predictions[i].y / count
    meanTargetX += targets[i].x / count
    meanTargetY += targets[i].y / count
  }

  // Symmetric XcᵀXc and the two columns of XcᵀYc.
  let xx = 0, xy = 0, yy = 0
  let rhs00 = 0, rhs01 = 0, rhs10 = 0, rhs11 = 0
  for (let i = 0; i < count; i++) {
    const x = predictions[i].x - meanX
    const y = predictions[i].y - meanY
    const targetX = targets[i].x - meanTargetX
    const targetY = targets[i].y - meanTargetY
    xx += x * x
    xy += x * y
    yy += y * y
    rhs00 += x * targetX
    rhs01 += x * targetY
    rhs10 += y * targetX
    rhs11 += y * targetY
  }

  const diagonalX = xx + lambda
  const diagonalY = yy + lambda
  if (![diagonalX, xy, diagonalY, rhs00, rhs01, rhs10, rhs11].every(Number.isFinite)) {
    throw new Error('Calibration arithmetic produced non-finite values')
  }

  // Scale the 2x2 solve to avoid squaring the magnitude of the Gram matrix
  // in its determinant. Scale the RHS by the same factor to preserve lambda.
  const scale = Math.max(diagonalX, diagonalY)
  const a = diagonalX / scale
  const b = xy / scale
  const c = diagonalY / scale
  const determinant = a * c - b * b
  if (!Number.isFinite(determinant) || determinant <= 16 * Number.EPSILON) {
    throw new Error('Calibration system is numerically singular; use positive lambda or more varied predictions')
  }
  const r00 = rhs00 / scale, r01 = rhs01 / scale
  const r10 = rhs10 / scale, r11 = rhs11 / scale
  const weight: CalibrationFit['weight'] = [
    [(c * r00 - b * r10) / determinant, (c * r01 - b * r11) / determinant],
    [(a * r10 - b * r00) / determinant, (a * r11 - b * r01) / determinant],
  ]
  const fit: CalibrationFit = {
    weight,
    intercept: [
      meanTargetX - meanX * weight[0][0] - meanY * weight[1][0],
      meanTargetY - meanX * weight[0][1] - meanY * weight[1][1],
    ],
    lambda,
  }
  validateFit(fit)
  return fit
}

/** Apply a fit once. Finite values outside [0, 1] are returned without clipping. */
export function applyCalibration(point: NormalizedPoint, fit: CalibrationFit): NormalizedPoint {
  validatePoint(point, 'point')
  validateFit(fit)
  const corrected = {
    x: point.x * fit.weight[0][0] + point.y * fit.weight[1][0] + fit.intercept[0],
    y: point.x * fit.weight[0][1] + point.y * fit.weight[1][1] + fit.intercept[1],
  }
  if (!Number.isFinite(corrected.x) || !Number.isFinite(corrected.y)) {
    throw new Error('Calibration application produced non-finite coordinates')
  }
  return corrected
}
