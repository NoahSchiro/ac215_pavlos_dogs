import { describe, expect, it } from 'vitest'
import { decideDrive, type DriveOptions, type ScreenRect } from './driveAssist'
import type { Direction } from './gazeDirection'
import { createGrid, generateMaze, type Maze, type MazePosition } from './mazeGenerator'

// 1000x1000 viewport; a 10x10 maze drawn at (200, 100), 600 px square, so one
// cell is 60 px = 0.06 of the height. Drive radius: 150 px (2.5 cells) across,
// 200 px (3.3 cells) tall. Pull ring out to 1.5x: 225 px across.
const VIEWPORT = { width: 1000, height: 1000 }
const RECT: ScreenRect = { left: 200, top: 100, width: 600, height: 600 }
const OPTIONS: DriveOptions = { radius: { x: 0.15, y: 0.2 }, pullScale: 1.5, dwelling: false, heldTarget: null, holdCells: 0.25 }
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

const decide = (maze: Maze, dCols: number, dRows: number, options: Partial<DriveOptions> = {}, player = CENTER) =>
  decideDrive(gazeAt(player, dCols, dRows), player, maze, RECT, VIEWPORT, { ...OPTIONS, ...options })

/** Walk `path` from `from`; the end cell, or null on hitting a wall or the grid edge. */
function walk(maze: Maze, from: MazePosition, path: Direction[]): MazePosition | null {
  let { row, col } = from
  for (const direction of path) {
    if (maze[row][col].walls[WALL[direction]]) return null
    row += STEP[direction][0]
    col += STEP[direction][1]
    if (!maze[row]?.[col]) return null
  }
  return { row, col }
}

describe('control zones around Pavlos', () => {
  it('stops while gaze stays inside his own cell', () => {
    expect(decide(openGrid(), 0.4, -0.3)).toEqual({ kind: 'stop' })
    expect(decide(openGrid(), -0.45, 0.45)).toEqual({ kind: 'stop' })
  })

  it('treats a glance outside the drive ellipse as exploring', () => {
    expect(decide(openGrid(), 2.6, 0)).toEqual({ kind: 'explore' }) // 156 px > 150 px across
    expect(decide(openGrid(), 2, 2.5)).toEqual({ kind: 'explore' }) // inside each axis, outside the ellipse
  })

  it('reaches further vertically than horizontally', () => {
    expect(decide(openGrid(), 0, 3)).toMatchObject({ kind: 'drive', path: ['down', 'down', 'down'] }) // 180 px < 200 px tall
  })

  it('moves with Pavlos: the same gaze point means different things from different cells', () => {
    const maze = openGrid()
    const gaze = gazeAt({ row: 5, col: 7 }, 0, 0)
    const from = (col: number) => decideDrive(gaze, { row: 5, col }, maze, RECT, VIEWPORT, OPTIONS).kind
    expect(from(5)).toBe('drive')
    expect(from(7)).toBe('stop')
    expect(from(1)).toBe('explore')
  })

  it('converts width fractions and height fractions separately on a wide screen', () => {
    const viewport = { width: 1600, height: 900 }
    const rect = { left: 500, top: 50, width: 600, height: 600 } // drive radius 135 px across = 2.25 cells
    const at = (dCols: number) =>
      decideDrive(gazeAt(CENTER, dCols, 0, viewport, rect), CENTER, openGrid(), rect, viewport, OPTIONS)
    expect(at(2)).toMatchObject({ kind: 'drive', path: ['right', 'right'] })
    expect(at(2.4)).toEqual({ kind: 'explore' })
  })
})

describe('snapping to a reachable cell', () => {
  it('drives straight along an open corridor', () => {
    const maze = carve(createGrid(10, 10), CENTER, ['right', 'right'])
    expect(decide(maze, 2.1, 0.2)).toEqual({ kind: 'drive', target: { row: 5, col: 7 }, path: ['right', 'right'], pulled: false })
  })

  it('snaps a look through a wall to the open cell nearest it', () => {
    const maze = carve(createGrid(10, 10), CENTER, ['right'])
    expect(decide(maze, 2, 0)).toMatchObject({ kind: 'drive', target: { row: 5, col: 6 }, path: ['right'] })
  })

  it('follows bends to the target', () => {
    const maze = carve(createGrid(10, 10), CENTER, ['up', 'right', 'right'])
    expect(decide(maze, 2, -1)).toMatchObject({ kind: 'drive', path: ['up', 'right', 'right'] })
  })

  it('drives diagonally across open space by a shortest path', () => {
    const decision = decide(openGrid(), 1, -1)
    expect(decision).toMatchObject({ kind: 'drive', target: { row: 4, col: 6 } })
    if (decision.kind === 'drive') expect(decision.path).toHaveLength(2)
  })

  it('stops when no open cell is nearer the gaze than his own', () => {
    expect(decide(createGrid(10, 10), 1, 0)).toEqual({ kind: 'stop' }) // walled in
  })

  it('only steers through cells inside the drive area', () => {
    // The right-hand neighbour is reachable only by a detour 4 rows up (240 px > 200 px).
    const maze = carve(createGrid(10, 10), CENTER, ['up', 'up', 'up', 'up', 'right', 'down', 'down', 'down', 'down'])
    expect(decide(maze, 1, 0)).toEqual({ kind: 'stop' })
  })

  it('never leaves the grid through the entrance gap', () => {
    const maze = openGrid()
    maze[0][0].walls.left = false
    expect(decide(maze, -1, 0, {}, { row: 0, col: 0 })).toEqual({ kind: 'stop' })
  })

  it('always returns an open shortest path to the target in a generated maze', () => {
    const rect = { left: 0, top: 0, width: 900, height: 900 }
    const viewport = { width: 900, height: 900 }
    const options = { ...OPTIONS, radius: { x: 0.25, y: 0.3 } }
    let drives = 0, bad = 0
    for (let trial = 0; trial < 10; trial++) {
      const maze = generateMaze(15, 15, { row: 0, col: 0 }, { row: 14, col: 14 })
      for (let row = 0; row < 15; row++) for (let col = 0; col < 15; col++) {
        for (let dr = -3; dr <= 3; dr++) for (let dc = -3; dc <= 3; dc++) {
          const gaze = gazeAt({ row, col }, dc, dr, viewport, rect, 15)
          const decision = decideDrive(gaze, { row, col }, maze, rect, viewport, options)
          if (decision.kind !== 'drive') continue
          drives++
          const end = walk(maze, { row, col }, decision.path)
          if (!end || end.row !== decision.target.row || end.col !== decision.target.col) bad++
        }
      }
    }
    expect(bad).toBe(0)
    expect(drives).toBeGreaterThan(0)
  })
})

describe('holding the target against jitter', () => {
  it('keeps the held target until another cell is clearly closer', () => {
    const held = { heldTarget: { row: 5, col: 6 } }
    // 1.55 cells right: column 7 is 0.1 cell closer than column 6, within the 0.25 hold.
    expect(decide(openGrid(), 1.55, 0)).toMatchObject({ target: { row: 5, col: 7 } })
    expect(decide(openGrid(), 1.55, 0, held)).toMatchObject({ target: { row: 5, col: 6 } })
    // 1.8 cells right: column 7 is 0.6 cell closer, so it takes over.
    expect(decide(openGrid(), 1.8, 0, held)).toMatchObject({ target: { row: 5, col: 7 } })
  })

  it('stays put against a small wobble toward a neighbour', () => {
    expect(decide(openGrid(), 0.6, 0)).toMatchObject({ kind: 'drive' })
    expect(decide(openGrid(), 0.6, 0, { heldTarget: CENTER })).toEqual({ kind: 'stop' })
  })
})

describe('the pull ring', () => {
  it('pulls toward a held stare just outside the drive ellipse', () => {
    expect(decide(openGrid(), 3.5, 0)).toEqual({ kind: 'explore' }) // 210 px: a glance does nothing
    expect(decide(openGrid(), 3.5, 0, { dwelling: true })).toEqual({
      kind: 'drive', target: { row: 5, col: 7 }, path: ['right', 'right'], pulled: true,
    })
  })

  it('ignores stares beyond the pull ring', () => {
    expect(decide(openGrid(), 4, 0, { dwelling: true })).toEqual({ kind: 'explore' }) // 240 px > 225 px
  })

  it('reaches further vertically, like the drive ellipse', () => {
    expect(decide(openGrid(), 0, -4.5, { dwelling: true })).toMatchObject({ kind: 'drive', path: ['up', 'up', 'up'], pulled: true })
  })

  it('still respects walls', () => {
    const maze = carve(createGrid(10, 10), CENTER, ['up'])
    expect(decide(maze, 3.5, 0, { dwelling: true })).toEqual({ kind: 'stop' })
  })
})

describe('invalid input', () => {
  const maze = openGrid()
  it.each([
    { name: 'no gaze', gaze: null, rect: RECT, viewport: VIEWPORT, options: OPTIONS },
    { name: 'NaN gaze', gaze: { x: NaN, y: 0.5 }, rect: RECT, viewport: VIEWPORT, options: OPTIONS },
    { name: 'unmeasured maze', gaze: { x: 0.5, y: 0.5 }, rect: { left: 0, top: 0, width: 0, height: 0 }, viewport: VIEWPORT, options: OPTIONS },
    { name: 'empty viewport', gaze: { x: 0.5, y: 0.5 }, rect: RECT, viewport: { width: 0, height: 0 }, options: OPTIONS },
    { name: 'pull ring inside the drive ellipse', gaze: { x: 0.5, y: 0.5 }, rect: RECT, viewport: VIEWPORT, options: { ...OPTIONS, pullScale: 0.5 } },
  ])('$name', ({ gaze, rect, viewport, options }) => {
    expect(decideDrive(gaze, CENTER, maze, rect, viewport, options)).toEqual({ kind: 'invalid' })
  })
})
