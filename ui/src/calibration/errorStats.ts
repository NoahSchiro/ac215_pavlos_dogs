import type { Viewport } from '../vision/viewport'
import type { NormalizedPoint } from './points'

/**
 * Gaze error broken down by axis, in fractions of the viewport height: the
 * same unit as GAZE_DRIVE_RADIUS, so both axes are comparable on screen.
 * Signs: +x means predictions land right of the target, +y means below it.
 */
export interface AxisError {
  /** Mean signed error: which way predictions lean. */
  bias: number
  /** Mean absolute error: the typical miss. */
  meanAbs: number
  /** 90th percentile absolute error: the miss to plan for. */
  p90Abs: number
}

export interface ErrorStats {
  x: AxisError
  y: AxisError
  frames: number
}

/** Mean signed error at one validation target, in viewport-height fractions. */
export interface TargetBias {
  target: NormalizedPoint
  frames: number
  dx: number
  dy: number
}

export interface Residual { target: NormalizedPoint; dx: number; dy: number }

/** Prediction minus target, with x rescaled from width to height fractions. */
export function residual(prediction: NormalizedPoint, target: NormalizedPoint, viewport: Viewport): Residual {
  return {
    target,
    dx: (prediction.x - target.x) * (viewport.width / viewport.height),
    dy: prediction.y - target.y,
  }
}

/** Nearest-rank percentile of non-negative values; 0 for none. */
export function percentile(values: number[], fraction: number): number {
  if (values.length === 0) return 0
  const sorted = [...values].sort((a, b) => a - b)
  return sorted[Math.max(0, Math.ceil(fraction * sorted.length) - 1)]
}

function axis(values: number[]): AxisError {
  const n = values.length || 1
  return {
    bias: values.reduce((sum, v) => sum + v, 0) / n,
    meanAbs: values.reduce((sum, v) => sum + Math.abs(v), 0) / n,
    p90Abs: percentile(values.map(Math.abs), 0.9),
  }
}

export function errorStats(residuals: Residual[]): ErrorStats {
  return {
    x: axis(residuals.map((r) => r.dx)),
    y: axis(residuals.map((r) => r.dy)),
    frames: residuals.length,
  }
}

export function biasByTarget(residuals: Residual[]): TargetBias[] {
  const groups = new Map<string, Residual[]>()
  for (const r of residuals) {
    const key = `${r.target.x},${r.target.y}`
    groups.set(key, [...(groups.get(key) ?? []), r])
  }
  return [...groups.values()].map((group) => ({
    target: group[0].target,
    frames: group.length,
    dx: group.reduce((sum, r) => sum + r.dx, 0) / group.length,
    dy: group.reduce((sum, r) => sum + r.dy, 0) / group.length,
  }))
}

/** How many cells past Pavlos the drive area should reach before error is added. */
export const DRIVE_REACH_CELLS = 2

export interface DriveSuggestion {
  /** Suggested GAZE_DRIVE_RADIUS, rounded up to 0.01. */
  radius: { x: number; y: number }
  /** Smallest cell (viewport-height fraction) whose half still covers the p90 miss on both axes. */
  minCell: number
}

/**
 * Size the drive area as reach plus the 90th-percentile miss on each axis,
 * and the cell so that looking straight at Pavlos usually stays in his cell.
 */
export function suggestDrive(stats: ErrorStats, cell: number): DriveSuggestion {
  const roundUp = (v: number) => Math.ceil(v * 100 - 1e-9) / 100
  return {
    radius: {
      x: roundUp(DRIVE_REACH_CELLS * cell + stats.x.p90Abs),
      y: roundUp(DRIVE_REACH_CELLS * cell + stats.y.p90Abs),
    },
    minCell: 2 * Math.max(stats.x.p90Abs, stats.y.p90Abs),
  }
}
