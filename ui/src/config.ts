/**
 * Deployment configuration, read at runtime rather than baked into the bundle.
 *
 * One built image runs both locally and on Cloud Run. What differs between
 * them -- which model to load, whether to send telemetry -- comes from
 * /config.json, which the nginx container renders from its environment at
 * startup (see ui/nginx/default.conf.template). Promoting a newly trained model
 * is then a redeploy with a different MODEL_URL, not a rebuild. Where there is
 * no config.json (the Vite dev server, the test runners) the defaults apply.
 */

export interface RuntimeConfig {
  /** Where the ONNX GazeNet is fetched from: same-origin, or any CORS-enabled URL. */
  modelUrl: string
  /** Beacon aggregate usage telemetry to /t on the serving origin. */
  telemetry: boolean
  /** The deployed build, reported with telemetry. */
  appVersion: string
}

export const DEFAULT_CONFIG: RuntimeConfig = {
  modelUrl: `${import.meta.env.BASE_URL}models/gazenet.onnx`,
  telemetry: false,
  appVersion: 'dev',
}

/** Fill anything missing or malformed in a fetched config with the defaults. */
export function parseConfig(raw: unknown): RuntimeConfig {
  const r = (typeof raw === 'object' && raw !== null ? raw : {}) as Record<string, unknown>
  const text = (v: unknown, fallback: string) => (typeof v === 'string' && v.trim() ? v.trim() : fallback)
  return {
    modelUrl: text(r.modelUrl, DEFAULT_CONFIG.modelUrl),
    telemetry: r.telemetry === true || r.telemetry === 'true',
    appVersion: text(r.appVersion, DEFAULT_CONFIG.appVersion),
  }
}

let cached: Promise<RuntimeConfig> | null = null

/** The runtime config, fetched once per page. Never rejects. */
export function loadRuntimeConfig(): Promise<RuntimeConfig> {
  cached ??= (async () => {
    try {
      const response = await fetch(`${import.meta.env.BASE_URL}config.json`, { cache: 'no-store' })
      // A dev server answers unknown paths with index.html, so a 200 that is
      // not JSON is as good as a 404 here.
      return response.ok ? parseConfig(await response.json()) : DEFAULT_CONFIG
    } catch {
      return DEFAULT_CONFIG
    }
  })()
  return cached
}
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
