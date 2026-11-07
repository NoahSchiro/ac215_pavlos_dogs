import { describe, expect, test } from 'vitest'
import { GazeKalmanFilter } from './kalman'

const FRAME_MS = 1000 / 30

// Seeded standard normal (xorshift32 + Box-Muller) so the tests are stable.
function makeNormal(seed: number) {
  let s = seed >>> 0 || 1
  const uniform = () => {
    s ^= s << 13
    s >>>= 0
    s ^= s >>> 17
    s ^= s << 5
    s >>>= 0
    return (s + 0.5) / 2 ** 32
  }
  return () => Math.sqrt(-2 * Math.log(uniform())) * Math.cos(2 * Math.PI * uniform())
}

function rms(values: number[]): number {
  return Math.sqrt(values.reduce((sum, v) => sum + v * v, 0) / values.length)
}

describe('GazeKalmanFilter', () => {
  test('returns null until the first measurement, then passes it through', () => {
    const kf = new GazeKalmanFilter()
    expect(kf.update(null, 0)).toBeNull()
    expect(kf.initialized).toBe(false)
    expect(kf.update({ x: 0.3, y: 0.7 }, 10)).toEqual({ x: 0.3, y: 0.7 })
    expect(kf.initialized).toBe(true)
  })

  test('reduces jitter on a fixation', () => {
    const normal = makeNormal(1)
    const kf = new GazeKalmanFilter({ measurementStd: 0.05 })
    const raw: number[] = []
    const filtered: number[] = []
    for (let i = 0; i < 90; i++) {
      const z = { x: 0.5 + 0.05 * normal(), y: 0.5 + 0.05 * normal() }
      const est = kf.update(z, i * FRAME_MS)!
      if (i >= 15) {
        raw.push(z.x - 0.5, z.y - 0.5)
        filtered.push(est.x - 0.5, est.y - 0.5)
      }
    }
    expect(rms(filtered)).toBeLessThan(0.6 * rms(raw))
  })

  test('follows smooth pursuit without drifting behind', () => {
    const normal = makeNormal(2)
    const kf = new GazeKalmanFilter()
    const speed = 0.3 // fractions per second
    const errors: number[] = []
    for (let i = 0; i < 60; i++) {
      const t = i * FRAME_MS
      const truth = 0.2 + (speed * t) / 1000
      const est = kf.update({ x: truth + 0.02 * normal(), y: 0.5 }, t)!
      if (i >= 20) errors.push(est.x - truth)
    }
    expect(Math.abs(errors.reduce((a, b) => a + b, 0) / errors.length)).toBeLessThan(0.02)
    expect(Math.abs(kf.velocity!.x - speed)).toBeLessThan(0.1)
  })

  test('ignores a single outlier', () => {
    const kf = new GazeKalmanFilter()
    for (let i = 0; i < 30; i++) kf.update({ x: 0.5, y: 0.5 }, i * FRAME_MS)
    const est = kf.update({ x: 0.95, y: 0.05 }, 30 * FRAME_MS)!
    expect(est.x).toBeCloseTo(0.5, 2)
    expect(est.y).toBeCloseTo(0.5, 2)
    // And the next normal frame is accepted again.
    expect(kf.update({ x: 0.5, y: 0.5 }, 31 * FRAME_MS)!.x).toBeCloseTo(0.5, 2)
  })

  test('jumps to the new point after a saccade', () => {
    const kf = new GazeKalmanFilter({ maxRejections: 2 })
    for (let i = 0; i < 30; i++) kf.update({ x: 0.2, y: 0.2 }, i * FRAME_MS)
    kf.update({ x: 0.8, y: 0.7 }, 30 * FRAME_MS)
    const est = kf.update({ x: 0.8, y: 0.7 }, 31 * FRAME_MS)!
    expect(est).toEqual({ x: 0.8, y: 0.7 })
  })

  test('coasts through dropped frames and resets after a long gap', () => {
    const kf = new GazeKalmanFilter({ maxGapMs: 500 })
    for (let i = 0; i < 30; i++) kf.update({ x: 0.4, y: 0.6 }, i * FRAME_MS)
    const coast = kf.update(null, 30 * FRAME_MS)!
    expect(coast.x).toBeCloseTo(0.4, 2)
    expect(kf.update(null, 29 * FRAME_MS + 600)).toBeNull()
    expect(kf.initialized).toBe(false)
    expect(kf.update({ x: 0.9, y: 0.1 }, 29 * FRAME_MS + 700)).toEqual({ x: 0.9, y: 0.1 })
  })

  test('clamps the estimate to the screen', () => {
    const kf = new GazeKalmanFilter({ gateChi2: Infinity })
    for (let i = 0; i < 10; i++) kf.update({ x: 0.9 + 0.02 * i, y: 0.5 }, i * FRAME_MS)
    const est = kf.update(null, 10 * FRAME_MS)!
    expect(est.x).toBeLessThanOrEqual(1)
    expect(est.x).toBeGreaterThan(0.9)
  })

  test('handles irregular frame timing and out-of-order timestamps', () => {
    const kf = new GazeKalmanFilter()
    const times = [0, 20, 70, 75, 140, 130, 200]
    for (const t of times) {
      const est = kf.update({ x: 0.5, y: 0.5 }, t)!
      expect(Number.isFinite(est.x) && Number.isFinite(est.y)).toBe(true)
      expect(est.x).toBeCloseTo(0.5, 6)
    }
  })
})
