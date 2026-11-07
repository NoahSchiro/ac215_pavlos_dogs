/**
 * Per-user ridge-regression gaze correction.
 *
 * A port of `fit_ridge` in model/src/training/calibration.py: an affine map
 * `y = x @ W + b` from the model's raw screen-fraction prediction `x` to where
 * the user was actually looking `y`. The penalty shrinks W toward zero and
 * leaves the intercept unpenalized. That beat shrinking toward the identity on
 * held-out MPII data, because this model's error is mostly noise rather than a
 * systematic per-user bias, and pulling noisy predictions toward the mean is
 * the better bet. Keep the two implementations in step; ridge.test.ts pins
 * this one to numbers produced by the Python one.
 */

/** A point in screen fractions: 0 is the left/top edge, 1 the right/bottom. */
export interface ScreenPoint {
  x: number
  y: number
}

export interface RidgeFit {
  /** 2x2 slope, row-major [w00, w01, w10, w11]: out.x = x*w00 + y*w10 + b.x */
  readonly weight: readonly [number, number, number, number]
  readonly intercept: readonly [number, number]
}

export function fitRidge(
  x: readonly ScreenPoint[],
  y: readonly ScreenPoint[],
  lamb = 0.1,
): RidgeFit {
  if (!(lamb >= 0)) throw new Error(`lamb must be >= 0, got ${lamb}`)
  if (x.length !== y.length) {
    throw new Error(`got ${x.length} inputs for ${y.length} targets`)
  }
  if (x.length === 0) throw new Error('cannot fit ridge regression on zero points')

  const n = x.length
  let xm0 = 0, xm1 = 0, ym0 = 0, ym1 = 0
  for (let i = 0; i < n; i++) {
    xm0 += x[i].x / n
    xm1 += x[i].y / n
    ym0 += y[i].x / n
    ym1 += y[i].y / n
  }

  // A = Xc^T Xc + lamb*I and B = Xc^T Yc, on the centered data, so the
  // intercept is fitted exactly and only the slope is penalized.
  let a00 = lamb, a01 = 0, a11 = lamb
  let b00 = 0, b01 = 0, b10 = 0, b11 = 0
  for (let i = 0; i < n; i++) {
    const dx0 = x[i].x - xm0, dx1 = x[i].y - xm1
    const dy0 = y[i].x - ym0, dy1 = y[i].y - ym1
    a00 += dx0 * dx0
    a01 += dx0 * dx1
    a11 += dx1 * dx1
    b00 += dx0 * dy0
    b01 += dx0 * dy1
    b10 += dx1 * dy0
    b11 += dx1 * dy1
  }

  const det = a00 * a11 - a01 * a01
  // With lamb > 0, A is positive definite and det > 0 always. It can only
  // vanish for lamb = 0 on inputs that do not span the plane.
  if (!(det > 0)) throw new Error('calibration inputs are degenerate')

  // W = A^-1 B, using the closed-form inverse of the symmetric 2x2 A.
  const w00 = (a11 * b00 - a01 * b10) / det
  const w01 = (a11 * b01 - a01 * b11) / det
  const w10 = (a00 * b10 - a01 * b00) / det
  const w11 = (a00 * b11 - a01 * b01) / det
  return {
    weight: [w00, w01, w10, w11],
    intercept: [ym0 - (xm0 * w00 + xm1 * w10), ym1 - (xm0 * w01 + xm1 * w11)],
  }
}

export function applyRidge(fit: RidgeFit, p: ScreenPoint): ScreenPoint {
  const [w00, w01, w10, w11] = fit.weight
  return {
    x: p.x * w00 + p.y * w10 + fit.intercept[0],
    y: p.x * w01 + p.y * w11 + fit.intercept[1],
  }
}
