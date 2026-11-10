/**
 * Game settings in one place.
 *
 * Change these values to tune the maze and gaze controls without hunting
 * through the components that use them.
 */

/** Maze size, in cells. */
export const MAZE_ROWS = 15
export const MAZE_COLS = 15

/** Pavlos's size, as a percentage of one maze cell. */
export const PLAYER_SIZE_PERCENT = 75

/** How often Pavlos moves one cell while an arrow key is held, in milliseconds. Lower is faster. */
export const KEYBOARD_MOVE_INTERVAL_MS = 160

/** How often Pavlos moves one cell while steered by gaze, in milliseconds. Lower is faster. */
export const GAZE_MOVE_INTERVAL_MS = 160

/** How often the webcam frame is run through the gaze model, in milliseconds. */
export const GAZE_PREDICTION_INTERVAL_MS = 200

/**
 * The "stop" zone in the middle of the screen, as screen fractions.
 * Looking between these values (on both axes) keeps Pavlos still.
 */
export const GAZE_CENTER_MIN = 0.4
export const GAZE_CENTER_MAX = 0.6

/**
 * Kalman smoothing of gaze predictions. Higher measurementStd or lower
 * accelerationStd gives smoother but slower-reacting gaze.
 */
export const GAZE_SMOOTHING = {
  measurementStd: 0.05,
  accelerationStd: 0.5,
}

/** How long the gaze crosshair takes to glide to each new position, in milliseconds. */
export const CROSSHAIR_GLIDE_MS = 450

/** Crosshair opacity, from 0 (invisible) to 1 (solid). */
export const CROSSHAIR_OPACITY = 0.55

/**
 * Address of the race backend (FastAPI). Set VITE_RACE_SERVER_URL when building
 * for production; defaults to a backend running locally on port 8000.
 */
export const RACE_SERVER_URL: string =
  (import.meta.env.VITE_RACE_SERVER_URL as string | undefined) ?? 'http://localhost:8000'
