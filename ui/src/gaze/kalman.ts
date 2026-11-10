import type { ScreenFraction } from '../vision/model'

export interface GazeKalmanOptions {
  measurementStd: number
  accelerationStd: number
  gateChi2: number
  maxRejections: number
  maxGapMs: number
}

export const DEFAULT_GAZE_KALMAN_OPTIONS: GazeKalmanOptions = {
  measurementStd: 0.05,
  accelerationStd: 0.5,
  gateChi2: 13.82,
  maxRejections: 2,
  maxGapMs: 500,
}

interface AxisState {
  p: number
  v: number
  pp: number
  pv: number
  vv: number
}

const INITIAL_VELOCITY_STD = 2

function initAxis(position: number, positionVar: number): AxisState {
  return {
    p: position,
    v: 0,
    pp: positionVar,
    pv: 0,
    vv: INITIAL_VELOCITY_STD ** 2,
  }
}

function predictAxis(s: AxisState, dt: number, q: number): AxisState {
  const dt2 = dt * dt
  const dt3 = dt2 * dt
  return {
    p: s.p + dt * s.v,
    v: s.v,
    pp: s.pp + 2 * dt * s.pv + dt2 * s.vv + (q * dt3) / 3,
    pv: s.pv + dt * s.vv + (q * dt2) / 2,
    vv: s.vv + q * dt,
  }
}

function updateAxis(s: AxisState, z: number, r: number): AxisState {
  const innovationVar = s.pp + r
  const kp = s.pp / innovationVar
  const kv = s.pv / innovationVar
  const y = z - s.p
  const a = 1 - kp
  return {
    p: s.p + kp * y,
    v: s.v + kv * y,
    pp: a * a * s.pp + kp * kp * r,
    pv: a * (s.pv - kv * s.pp) + kp * kv * r,
    vv: s.vv - 2 * kv * s.pv + kv * kv * s.pp + kv * kv * r,
  }
}

export class GazeKalmanFilter {
  readonly options: GazeKalmanOptions
  private x: AxisState | null = null
  private y: AxisState | null = null
  private lastTimeMs = 0
  private lastAcceptedMs = 0
  private rejections = 0

  constructor(options: Partial<GazeKalmanOptions> = {}) {
    this.options = { ...DEFAULT_GAZE_KALMAN_OPTIONS, ...options }
  }

  get initialized(): boolean {
    return this.x !== null
  }

  get velocity(): ScreenFraction | null {
    if (!this.x || !this.y) return null
    return { x: this.x.v, y: this.y.v }
  }

  reset(): void {
    this.x = null
    this.y = null
    this.rejections = 0
  }

  update(measurement: ScreenFraction | null, timestampMs: number): ScreenFraction | null {
    const { measurementStd, accelerationStd, gateChi2, maxRejections, maxGapMs } = this.options
    const r = measurementStd ** 2
    const q = accelerationStd ** 2

    if (this.x && timestampMs - this.lastAcceptedMs > maxGapMs) this.reset()

    if (!this.x || !this.y) {
      if (!measurement) return null
      this.start(measurement, timestampMs, r)
      return this.estimate()
    }

    const dt = Math.max(0, (timestampMs - this.lastTimeMs) / 1000)
    this.lastTimeMs = Math.max(this.lastTimeMs, timestampMs)
    if (dt > 0) {
      this.x = predictAxis(this.x, dt, q)
      this.y = predictAxis(this.y, dt, q)
    }

    if (measurement) {
      const ex = measurement.x - this.x.p
      const ey = measurement.y - this.y.p
      const d2 = (ex * ex) / (this.x.pp + r) + (ey * ey) / (this.y.pp + r)
      if (d2 <= gateChi2) {
        this.x = updateAxis(this.x, measurement.x, r)
        this.y = updateAxis(this.y, measurement.y, r)
        this.lastAcceptedMs = timestampMs
        this.rejections = 0
      } else if (++this.rejections >= maxRejections) {
        this.start(measurement, timestampMs, r)
      }
    }

    return this.estimate()
  }

  private start(measurement: ScreenFraction, timestampMs: number, r: number): void {
    this.x = initAxis(measurement.x, r)
    this.y = initAxis(measurement.y, r)
    this.lastTimeMs = timestampMs
    this.lastAcceptedMs = timestampMs
    this.rejections = 0
  }

  private estimate(): ScreenFraction | null {
    if (!this.x || !this.y) return null
    const clamp = (v: number) => Math.min(1, Math.max(0, v))
    return { x: clamp(this.x.p), y: clamp(this.y.p) }
  }
}
