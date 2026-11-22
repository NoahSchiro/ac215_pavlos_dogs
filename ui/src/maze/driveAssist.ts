import type { Maze, MazePosition } from './mazeGenerator'
import type { Direction } from './gazeDirection'
import type { ScreenFraction } from '../vision/model'
import type { Viewport } from '../vision/viewport'

/** The maze's on-screen box, in CSS pixels (as from getBoundingClientRect). */
export interface ScreenRect { left: number; top: number; width: number; height: number }

/** Drive ellipse half-widths, as fractions of the viewport height. */
export interface DriveRadius { x: number; y: number }

/** Gaze target relative to Pavlos, in whole cells (+col is right, +row is down). */
export interface CellOffset { cols: number; rows: number }

export type DriveDecision =
  /** Missing or unusable gaze, viewport or maze geometry. */
  | { kind: 'invalid' }
  /** Gaze outside the drive ellipse: the user is looking around. */
  | { kind: 'explore' }
  /** Gaze inside Pavlos's own cell. */
  | { kind: 'stop' }
  /** Target cell has no open path of one or two straight legs, or more than one. */
  | { kind: 'blocked'; offset: CellOffset; reason: 'walls' | 'ambiguous' }
  /** Take `path[0]` now; `path` is the whole route to the target cell. */
  | { kind: 'drive'; offset: CellOffset; path: Direction[] }

const STEP: Record<Direction, { rows: number; cols: number }> = {
  up: { rows: -1, cols: 0 },
  down: { rows: 1, cols: 0 },
  left: { rows: 0, cols: -1 },
  right: { rows: 0, cols: 1 },
}

const WALL: Record<Direction, 'top' | 'bottom' | 'left' | 'right'> = {
  up: 'top', down: 'bottom', left: 'left', right: 'right',
}

const positive = (n: number) => Number.isFinite(n) && n > 0

/**
 * Decide how gaze steers Pavlos, relative to where he is now.
 *
 * The gaze point is a viewport fraction (the model's output). It is measured
 * from Pavlos's centre in pixels, so the control zones move with him:
 *   - outside the drive ellipse -> explore (no move);
 *   - within half a cell of him on both axes -> stop;
 *   - otherwise the nearest cell is the target, reached by at most two straight
 *     legs (horizontal then vertical, or vertical then horizontal). It drives
 *     only when exactly one of those routes is open: no wall crossings, no
 *     leaving the grid.
 */
export function decideDrive(
  gaze: ScreenFraction | null | undefined,
  player: MazePosition,
  maze: Maze,
  mazeRect: ScreenRect,
  viewport: Viewport,
  radius: DriveRadius,
): DriveDecision {
  const rows = maze.length
  const cols = maze[0]?.length ?? 0
  if (!gaze || !Number.isFinite(gaze.x) || !Number.isFinite(gaze.y) ||
      !positive(viewport.width) || !positive(viewport.height) ||
      !positive(mazeRect.width) || !positive(mazeRect.height) ||
      !Number.isFinite(mazeRect.left) || !Number.isFinite(mazeRect.top) ||
      !positive(radius.x) || !positive(radius.y) || rows === 0 || cols === 0) {
    return { kind: 'invalid' }
  }

  const cellWidth = mazeRect.width / cols
  const cellHeight = mazeRect.height / rows
  const dx = gaze.x * viewport.width - (mazeRect.left + (player.col + 0.5) * cellWidth)
  const dy = gaze.y * viewport.height - (mazeRect.top + (player.row + 0.5) * cellHeight)

  const rx = radius.x * viewport.height
  const ry = radius.y * viewport.height
  if ((dx / rx) ** 2 + (dy / ry) ** 2 > 1) return { kind: 'explore' }

  // `+ 0` turns Math.round's -0 into 0.
  const offset = { cols: Math.round(dx / cellWidth) + 0, rows: Math.round(dy / cellHeight) + 0 }
  if (offset.cols === 0 && offset.rows === 0) return { kind: 'stop' }

  const horizontal = legs(offset.cols < 0 ? 'left' : 'right', Math.abs(offset.cols))
  const vertical = legs(offset.rows < 0 ? 'up' : 'down', Math.abs(offset.rows))
  const routes = offset.cols === 0 || offset.rows === 0
    ? [[...horizontal, ...vertical]]
    : [[...horizontal, ...vertical], [...vertical, ...horizontal]]
  const open = routes.filter((route) => routeIsOpen(maze, player, route))

  if (open.length === 1) return { kind: 'drive', offset, path: open[0] }
  return { kind: 'blocked', offset, reason: open.length === 0 ? 'walls' : 'ambiguous' }
}

function legs(direction: Direction, count: number): Direction[] {
  return Array.from({ length: count }, () => direction)
}

function routeIsOpen(maze: Maze, start: MazePosition, route: Direction[]): boolean {
  let { row, col } = start
  for (const direction of route) {
    const cell = maze[row]?.[col]
    if (!cell || cell.walls[WALL[direction]]) return false
    row += STEP[direction].rows
    col += STEP[direction].cols
    // The entrance and exit gaps lead off the grid; Pavlos cannot go there.
    if (row < 0 || row >= maze.length || col < 0 || col >= maze[0].length) return false
  }
  return true
}
