import type { Maze, MazePosition } from './mazeGenerator'
import type { Direction } from './gazeDirection'
import type { ScreenFraction } from '../vision/model'
import type { Viewport } from '../vision/viewport'

/** The maze's on-screen box, in CSS pixels (as from getBoundingClientRect). */
export interface ScreenRect { left: number; top: number; width: number; height: number }

/** Drive ellipse half-widths, as fractions of the viewport height. */
export interface DriveRadius { x: number; y: number }

export interface DriveOptions {
  /** The drive ellipse around Pavlos: gaze here steers him straight away. */
  radius: DriveRadius
  /** Outer pull ellipse, as a multiple of `radius` (>= 1): a held stare here pulls him. */
  pullScale: number
  /** True once the gaze has held still long enough to count as a stare. */
  dwelling: boolean
  /** Last tick's target. It is kept unless another cell is clearly closer to the gaze. */
  heldTarget?: MazePosition | null
  /** How much closer, in cells, another cell must be to replace `heldTarget`. */
  holdCells: number
}

export type DriveDecision =
  /** Missing or unusable gaze, viewport, maze geometry or options. */
  | { kind: 'invalid' }
  /** Gaze outside the drive ellipse and not a held stare in the pull ring: looking around. */
  | { kind: 'explore' }
  /** Pavlos's own cell is the reachable cell nearest the gaze. */
  | { kind: 'stop' }
  /** Take `path[0]` now; `path` is the shortest route to `target`. `pulled`: from a stare in the pull ring. */
  | { kind: 'drive'; target: MazePosition; path: Direction[]; pulled: boolean }

const STEP: Record<Direction, { rows: number; cols: number }> = {
  up: { rows: -1, cols: 0 },
  down: { rows: 1, cols: 0 },
  left: { rows: 0, cols: -1 },
  right: { rows: 0, cols: 1 },
}

const WALL: Record<Direction, 'top' | 'bottom' | 'left' | 'right'> = {
  up: 'top', down: 'bottom', left: 'left', right: 'right',
}

const DIRECTIONS: Direction[] = ['right', 'left', 'down', 'up']

const positive = (n: number) => Number.isFinite(n) && n > 0
const samePosition = (a: MazePosition, b: MazePosition) => a.row === b.row && a.col === b.col

/**
 * Decide how gaze steers Pavlos, relative to where he is now.
 *
 * The gaze point is a viewport fraction (the model's output), measured from
 * Pavlos's centre in pixels, so every zone moves with him:
 *   - inside the drive ellipse -> steer;
 *   - in the pull ring around it -> steer only while the user holds a stare
 *     (gaze error often lands a look just outside the ellipse);
 *   - further out -> explore (no move).
 * Steering snaps to the cell Pavlos can actually reach that is nearest the
 * gaze. Reachable means connected to him through open walls, without leaving
 * the drive ellipse. Vertical distance counts less, because vertical gaze is
 * less accurate. He then follows the shortest path there; in a generated maze
 * that path is the only one. If his own cell is nearest, he stops.
 */
export function decideDrive(
  gaze: ScreenFraction | null | undefined,
  player: MazePosition,
  maze: Maze,
  mazeRect: ScreenRect,
  viewport: Viewport,
  options: DriveOptions,
): DriveDecision {
  const { radius, pullScale, dwelling, heldTarget, holdCells } = options
  const rows = maze.length
  const cols = maze[0]?.length ?? 0
  if (!gaze || !Number.isFinite(gaze.x) || !Number.isFinite(gaze.y) ||
      !positive(viewport.width) || !positive(viewport.height) ||
      !positive(mazeRect.width) || !positive(mazeRect.height) ||
      !Number.isFinite(mazeRect.left) || !Number.isFinite(mazeRect.top) ||
      !positive(radius.x) || !positive(radius.y) || rows === 0 || cols === 0 ||
      !(pullScale >= 1) || !Number.isFinite(pullScale) || !(holdCells >= 0)) {
    return { kind: 'invalid' }
  }

  const cellWidth = mazeRect.width / cols
  const cellHeight = mazeRect.height / rows
  const centre = (p: MazePosition) => ({
    x: mazeRect.left + (p.col + 0.5) * cellWidth,
    y: mazeRect.top + (p.row + 0.5) * cellHeight,
  })
  const origin = centre(player)
  const rx = radius.x * viewport.height
  const ry = radius.y * viewport.height
  // Squared ellipse distance from Pavlos: 1 is the drive ellipse's edge.
  const reach = (x: number, y: number) => ((x - origin.x) / rx) ** 2 + ((y - origin.y) / ry) ** 2

  const gx = gaze.x * viewport.width
  const gy = gaze.y * viewport.height
  const gazeReach = reach(gx, gy)
  const pulled = gazeReach > 1
  if (pulled && !(dwelling && gazeReach <= pullScale ** 2)) return { kind: 'explore' }

  const candidates = reachableCells(maze, player, (p) => {
    const c = centre(p)
    return reach(c.x, c.y) <= 1 + 1e-9
  })

  // Pixels to the gaze, with vertical distance shrunk by the ellipse's aspect.
  const squash = rx / ry
  const score = (p: MazePosition) => {
    const c = centre(p)
    return Math.hypot(gx - c.x, (gy - c.y) * squash)
  }

  let best = candidates[0]
  let bestScore = score(best.position)
  for (const candidate of candidates.slice(1)) {
    const s = score(candidate.position)
    if (s < bestScore) {
      best = candidate
      bestScore = s
    }
  }

  // Hysteresis: a jittery gaze between two cells keeps the current target.
  const held = heldTarget && candidates.find((c) => samePosition(c.position, heldTarget))
  if (held && score(held.position) <= bestScore + holdCells * cellWidth) best = held

  if (best.path.length === 0) return { kind: 'stop' }
  return { kind: 'drive', target: best.position, path: best.path, pulled }
}

interface Reachable { position: MazePosition; path: Direction[] }

/** Breadth-first search from `start` through open walls, staying in the grid and `inside`. */
function reachableCells(maze: Maze, start: MazePosition, inside: (p: MazePosition) => boolean): Reachable[] {
  const rows = maze.length
  const cols = maze[0].length
  const seen = new Set([start.row * cols + start.col])
  const found: Reachable[] = [{ position: start, path: [] }]
  for (let i = 0; i < found.length; i++) {
    const { position, path } = found[i]
    const cell = maze[position.row][position.col]
    for (const direction of DIRECTIONS) {
      if (cell.walls[WALL[direction]]) continue
      const next = { row: position.row + STEP[direction].rows, col: position.col + STEP[direction].cols }
      // The entrance and exit gaps lead off the grid; Pavlos cannot go there.
      if (next.row < 0 || next.row >= rows || next.col < 0 || next.col >= cols) continue
      const key = next.row * cols + next.col
      if (seen.has(key) || !inside(next)) continue
      seen.add(key)
      found.push({ position: next, path: [...path, direction] })
    }
  }
  return found
}

/** Where the gaze is relative to Pavlos. */
export interface GazeRelation {
  /** Squared ellipse distance from his centre: 1 is the drive ellipse's edge. */
  reach: number
  /** Inside his own cell: the user is looking at him (stopping on purpose). */
  inOwnCell: boolean
}

/** Measure the gaze against Pavlos, as decideDrive does; null for unusable input. */
export function gazeRelation(
  gaze: ScreenFraction | null | undefined,
  player: MazePosition,
  maze: Maze,
  mazeRect: ScreenRect,
  viewport: Viewport,
  radius: DriveRadius,
): GazeRelation | null {
  const rows = maze.length
  const cols = maze[0]?.length ?? 0
  if (!gaze || !Number.isFinite(gaze.x) || !Number.isFinite(gaze.y) ||
      !positive(viewport.width) || !positive(viewport.height) ||
      !positive(mazeRect.width) || !positive(mazeRect.height) ||
      !positive(radius.x) || !positive(radius.y) || rows === 0 || cols === 0) {
    return null
  }
  const cellWidth = mazeRect.width / cols
  const cellHeight = mazeRect.height / rows
  const dx = gaze.x * viewport.width - (mazeRect.left + (player.col + 0.5) * cellWidth)
  const dy = gaze.y * viewport.height - (mazeRect.top + (player.row + 0.5) * cellHeight)
  return {
    reach: (dx / (radius.x * viewport.height)) ** 2 + (dy / (radius.y * viewport.height)) ** 2,
    inOwnCell: Math.abs(dx) <= cellWidth / 2 && Math.abs(dy) <= cellHeight / 2,
  }
}
