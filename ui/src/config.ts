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


/** Game settings */

/**
 * Maze size, in cells. Cells are square and big because gaze is imprecise:
 * Pavlos only holds still while gaze stays inside his own cell. Wider than
 * tall because horizontal gaze is more accurate than vertical.
 */
export const MAZE_ROWS = 9
export const MAZE_COLS = 13

/** Maximum on-screen maze width: the full height left after the status line, unless the window is too narrow. */
export const MAZE_DISPLAY_SIZE = `min(94vw, calc((100dvh - 100px) * ${MAZE_COLS} / ${MAZE_ROWS}))`

/** One maze cell as a fraction of the viewport height. Keep in step with MAZE_DISPLAY_SIZE. */
export function mazeCellFraction(viewport: { width: number; height: number }): number {
  const mazeHeightPx = Math.min((0.94 * viewport.width * MAZE_ROWS) / MAZE_COLS, viewport.height - 100)
  return Math.max(0, mazeHeightPx) / MAZE_ROWS / viewport.height
}

/** Pavlos's size, as a percentage of one maze cell. */
export const PLAYER_SIZE_PERCENT = 75

/** How often Pavlos moves one cell while an arrow key is held, in milliseconds. Lower is faster. */
export const KEYBOARD_MOVE_INTERVAL_MS = 160

/** How often Pavlos moves one cell while steered by gaze, in milliseconds. Lower is faster. */
export const GAZE_MOVE_INTERVAL_MS = 160

/** How often the webcam frame is run through the gaze model, in milliseconds. */
export const GAZE_PREDICTION_INTERVAL_MS = 200

/**
 * Where gaze can steer Pavlos: an ellipse centred on him, with these
 * half-widths as fractions of the screen height (so a circle on screen when
 * x === y). Gaze here snaps to the nearest cell he can reach and drives him
 * there; outside it, only a held stare in the pull ring moves him. Taller than
 * wide because vertical gaze predictions are less accurate. One maze cell is
 * about 0.1 of the height, so this reaches about 2.5 cells sideways and 3
 * cells up or down.
 */
export const GAZE_DRIVE_RADIUS = { x: 0.25, y: 0.3 }

/**
 * The pull ring: out to this multiple of GAZE_DRIVE_RADIUS, a held stare
 * (GAZE_DWELL_MS within GAZE_DWELL_RADIUS) pulls Pavlos toward it. Gaze error
 * often lands a look at a nearby cell just outside the drive ellipse.
 */
export const GAZE_PULL_SCALE = 1.5

/** How long gaze must hold still to count as a stare. */
export const GAZE_DWELL_MS = 600

/** Gaze within this distance of the stare's mean (fraction of the screen height) counts as holding still. */
export const GAZE_DWELL_RADIUS = 0.08

/** Keep the current target cell unless another reachable cell is this many cells closer to the gaze. */
export const GAZE_TARGET_HOLD_CELLS = 0.25

/**
 * Dynamic recentering: stare at Pavlos to fix a constant gaze offset. It is
 * started from the Recenter strip (RECENTER_STRIP). A steady stare near him
 * for RECENTER.confirmMs then starts capture; the median of the next
 * captureSamples predictions (one per GAZE_PREDICTION_INTERVAL_MS, so about
 * 1 s) is compared with his centre, and that shift is added to later gaze.
 * Stares further than maxShift (fraction of the screen height) from him are
 * taken as looking elsewhere. Pavlos holds still meanwhile.
 */
export const RECENTER = {
  confirmMs: 1000,
  captureSamples: 5,
  holdRadius: 0.08,
  maxShift: 0.25,
  timeoutMs: 10_000,
}

/** How long Pavlos shows the recenter result before play resumes. */
export const RECENTER_RESULT_MS = 1000

/**
 * The Recenter strip beside the maze, started by looking at it for dwellMs.
 * Its hit zone runs from the strip's column past the right screen edge, and
 * `pad` (fraction of the screen height) above and below the strip, so only
 * horizontal gaze accuracy matters. Up to maxMisses stray samples in a row
 * are forgiven.
 */
export const RECENTER_STRIP = { dwellMs: 1600, pad: 0.04, maxMisses: 1 }

/** Outline the drive area around Pavlos while gaze is steering him. */
export const SHOW_DRIVE_AREA = true

/**
 * Legacy screen-centre steering, still used by the race mode branch.
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
