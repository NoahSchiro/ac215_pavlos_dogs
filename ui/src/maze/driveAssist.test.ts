import { describe, expect, it } from 'vitest'
import { decideDrive, type ScreenRect } from './driveAssist'
import type { Direction } from './gazeDirection'
import { createGrid, generateMaze, type Maze, type MazePosition } from './mazeGenerator'

// 1000x1000 viewport; a 10x10 maze drawn at (200, 100), 600 px square, so one
// cell is 60 px = 0.06 of the height. Drive radius: 150 px across, 200 px tall.
const VIEWPORT = { width: 1000, height: 1000 }
const RECT: ScreenRect = { left: 200, top: 100, width: 600, height: 600 }
const RADIUS = { x: 0.15, y: 0.2 }
const CENTER = { row: 5, col: 5 }

const OPPOSITE: Record<Direction, Direction> = { up: 'down', down: 'up', left: 'right', right: 'left' }
const WALL = { up: 'top', down: 'bottom', left: 'left', right: 'right' } as const
const STEP = { up: [-1, 0], down: [1, 0], left: [0, -1], right: [0, 1] } as const

/** Knock down walls along a route starting at `from`. */
function carve(maze: Maze, from: MazePosition, route: Direction[]): Maze {
  let { row, col } = from
  for (const direction of route) {
    const [dr, dc] = STEP[direction]
    maze[row][col].walls[WALL[direction]] = false
    row += dr
    col += dc
    maze[row][col].walls[WALL[OPPOSITE[direction]]] = false
  }
  return maze
}

/** No interior walls at all. */
function openGrid(rows = 10, cols = 10): Maze {
  const maze = createGrid(rows, cols)
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
    if (c + 1 < cols) carve(maze, { row: r, col: c }, ['right'])
    if (r + 1 < rows) carve(maze, { row: r, col: c }, ['down'])
  }
  return maze
}

/** Gaze `dCols`/`dRows` cells away from the centre of `player`, as viewport fractions. */
function gazeAt(player: MazePosition, dCols: number, dRows: number, viewport = VIEWPORT, rect = RECT, size = 10) {
  return {
    x: (rect.left + (player.col + 0.5 + dCols) * (rect.width / size)) / viewport.width,
    y: (rect.top + (player.row + 0.5 + dRows) * (rect.height / size)) / viewport.height,
  }
}

const decide = (maze: Maze, dCols: number, dRows: number, player = CENTER) =>
  decideDrive(gazeAt(player, dCols, dRows), player, maze, RECT, VIEWPORT, RADIUS)

describe('control zones around Pavlos', () => {
  it('stops while gaze stays inside his own cell', () => {
    expect(decide(openGrid(), 0.4, -0.3)).toEqual({ kind: 'stop' })
    expect(decide(openGrid(), -0.45, 0.45)).toEqual({ kind: 'stop' })
  })

  it('treats gaze outside the drive ellipse as exploring', () => {
    expect(decide(openGrid(), 2.6, 0)).toEqual({ kind: 'explore' }) // 156 px > 150 px across
    expect(decide(openGrid(), 2, 2.5)).toEqual({ kind: 'explore' }) // inside each axis, outside the ellipse
  })

  it('reaches further vertically than horizontally', () => {
    expect(decide(openGrid(), 0, 3)).toMatchObject({ kind: 'drive', path: ['down', 'down', 'down'] }) // 180 px < 200 px tall
  })

  it('moves with Pavlos: the same gaze point means different things from different cells', () => {
    const maze = openGrid()
    const gaze = gazeAt({ row: 5, col: 7 }, 0, 0)
    const from = (col: number) => decideDrive(gaze, { row: 5, col }, maze, RECT, VIEWPORT, RADIUS).kind
    expect(from(5)).toBe('drive')
    expect(from(7)).toBe('stop')
    expect(from(1)).toBe('explore')
  })

  it('converts width fractions and height fractions separately on a wide screen', () => {
    const viewport = { width: 1600, height: 900 }
    const rect = { left: 500, top: 50, width: 600, height: 600 } // drive radius 135 px across = 2.25 cells
    const at = (dCols: number) =>
      decideDrive(gazeAt(CENTER, dCols, 0, viewport, rect), CENTER, openGrid(), rect, viewport, RADIUS)
    expect(at(2)).toMatchObject({ kind: 'drive', path: ['right', 'right'] })
    expect(at(2.4)).toEqual({ kind: 'explore' })
  })
})

describe('one- and two-leg paths', () => {
  it('drives straight along an open corridor', () => {
    const maze = carve(createGrid(10, 10), CENTER, ['right', 'right'])
    expect(decide(maze, 2.1, 0.2)).toEqual({ kind: 'drive', offset: { cols: 2, rows: 0 }, path: ['right', 'right'] })
  })

  it('does not move toward a target behind a wall', () => {
    const maze = carve(createGrid(10, 10), CENTER, ['right'])
    expect(decide(maze, 2, 0)).toEqual({ kind: 'blocked', offset: { cols: 2, rows: 0 }, reason: 'walls' })
  })

  it('takes the only open L: horizontal first', () => {
    const maze = carve(createGrid(10, 10), CENTER, ['right', 'up'])
    expect(decide(maze, 1, -1)).toMatchObject({ kind: 'drive', path: ['right', 'up'] })
  })

  it('takes the only open L: vertical first', () => {
    const maze = carve(createGrid(10, 10), CENTER, ['up', 'right', 'right'])
    expect(decide(maze, 2, -1)).toMatchObject({ kind: 'drive', path: ['up', 'right', 'right'] })
  })

  it('refuses when both L orders are open (not unique)', () => {
    expect(decide(openGrid(), 1, -1)).toEqual({ kind: 'blocked', offset: { cols: 1, rows: -1 }, reason: 'ambiguous' })
  })

  it('never leaves the grid through the entrance gap', () => {
    const maze = openGrid()
    const corner = { row: 0, col: 0 }
    maze[0][0].walls.left = false
    expect(decide(maze, -1, 0, corner)).toMatchObject({ kind: 'blocked', reason: 'walls' })
  })

  it('never finds two open L routes in a generated maze (one path between any two cells)', () => {
    const rect = { left: 0, top: 0, width: 900, height: 900 }
    const viewport = { width: 900, height: 900 }
    let ambiguous = 0, drives = 0
    for (let trial = 0; trial < 10; trial++) {
      const maze = generateMaze(15, 15, { row: 0, col: 0 }, { row: 14, col: 14 })
      for (let row = 0; row < 15; row++) for (let col = 0; col < 15; col++) {
        for (let dr = -3; dr <= 3; dr++) for (let dc = -3; dc <= 3; dc++) {
          const gaze = gazeAt({ row, col }, dc, dr, viewport, rect, 15)
          const decision = decideDrive(gaze, { row, col }, maze, rect, viewport, { x: 1, y: 1 })
          if (decision.kind === 'blocked' && decision.reason === 'ambiguous') ambiguous++
          if (decision.kind === 'drive') drives++
        }
      }
    }
    expect(ambiguous).toBe(0)
    expect(drives).toBeGreaterThan(0)
  })
})

describe('invalid input', () => {
  const maze = openGrid()
  it.each([
    { name: 'no gaze', gaze: null, rect: RECT, viewport: VIEWPORT },
    { name: 'NaN gaze', gaze: { x: NaN, y: 0.5 }, rect: RECT, viewport: VIEWPORT },
    { name: 'unmeasured maze', gaze: { x: 0.5, y: 0.5 }, rect: { left: 0, top: 0, width: 0, height: 0 }, viewport: VIEWPORT },
    { name: 'empty viewport', gaze: { x: 0.5, y: 0.5 }, rect: RECT, viewport: { width: 0, height: 0 } },
  ])('$name', ({ gaze, rect, viewport }) => {
    expect(decideDrive(gaze, CENTER, maze, rect, viewport, RADIUS)).toEqual({ kind: 'invalid' })
  })
})
