/**
 * Calibration targets and the data captured at each one.
 *
 * All screen positions are normalized to [0, 1] on both axes, with (0, 0)
 * at the top-left corner of the viewport. This is the same label space the
 * gaze model predicts in, so samples can be fed to per-user fitting directly.
 */

import type { CaptureCamera } from '../vision/camera'

export interface NormalizedPoint {
  /** Horizontal position as a fraction of viewport width, 0 = left. */
  readonly x: number
  /** Vertical position as a fraction of viewport height, 0 = top. */
  readonly y: number
}

/** One captured frame paired with the target the user was looking at. */
export interface CalibrationSample {
  /** Full webcam frame, JPEG encoded. */
  readonly image: Blob
  /** Target the user was asked to look at. */
  readonly target: NormalizedPoint
  /** Index of the target in the active capture grid. */
  readonly pointIndex: number
  /** Viewport size in CSS pixels when the frame was captured. */
  readonly viewportWidth: number
  readonly viewportHeight: number
  /** Capture time, milliseconds since the Unix epoch. */
  readonly timestamp: number
}

/** The output of one complete calibration run. */
export interface CalibrationResult {
  readonly samples: readonly CalibrationSample[]
  /** Actual capture track settings; absence prevents verified persistence/reuse. */
  readonly camera?: CaptureCamera | null
  /** Completion time, milliseconds since the Unix epoch. */
  readonly completedAt: number
}

/** Inset from each edge so targets stay clear of the physical bezel. */
const EDGE_INSET = 0.1

const GRID_COORDS = [EDGE_INSET, 0.5, 1 - EDGE_INSET] as const

/**
 * Nine targets on a 3x3 grid, ordered left to right, then top to bottom.
 * Index 0 is top-left, 4 is the center, 8 is bottom-right.
 */
export const CALIBRATION_POINTS: readonly NormalizedPoint[] = GRID_COORDS.flatMap(
  (y) => GRID_COORDS.map((x) => ({ x, y })),
)

/** Independent targets: validation frames must never be used for fitting. */
export const VALIDATION_POINTS: readonly NormalizedPoint[] = [
  { x: 0.25, y: 0.25 }, { x: 0.75, y: 0.25 },
  { x: 0.25, y: 0.75 }, { x: 0.75, y: 0.75 },
]

/** Frames captured per target on each Space press. */
export const FRAMES_PER_POINT = 3

/** Delay between frames within one burst. */
export const FRAME_INTERVAL_MS = 100
