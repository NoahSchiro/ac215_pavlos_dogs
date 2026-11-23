import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { act, fireEvent, render, screen } from '@testing-library/react'
import Maze from './Maze'
import { generateMaze } from './mazeGenerator'
import { GAZE_DRIVE_RADIUS, GAZE_DWELL_MS, GAZE_PULL_SCALE, MAZE_COLS, MAZE_ROWS, RECENTER_RESULT_MS } from '../config'

// Mock the maze generator to create a predictable maze for testing.
vi.mock('./mazeGenerator', () => ({
  generateMaze: vi.fn(),
}))

const mockedGenerateMaze = vi.mocked(generateMaze)

const ROWS = MAZE_ROWS
const COLS = MAZE_COLS

// Create a maze with no internal walls, but closed outer boundaries.
function createTestMaze() {
  return Array.from({ length: ROWS }, (_, row) =>
    Array.from({ length: COLS }, (_, col) => ({
      row,
      col,
      visited: true,
      walls: {
        top: row === 0,
        right: col === COLS - 1,
        bottom: row === ROWS - 1,
        left: col === 0,
      },
    })),
  )
}

// Render the maze with predictable walls.
function setupMaze() {
  mockedGenerateMaze.mockImplementation(createTestMaze)
  render(<Maze />)
}

// Simulate pressing and releasing an arrow key.
function pressArrow(key: string) {
  fireEvent.keyDown(window, { key })
  fireEvent.keyUp(window, { key })
}

// Convert a player's CSS position back into a grid coordinate.
function getPlayerPosition() {
  const player = document.querySelector('.maze-player') as HTMLElement

  const col = Math.round((parseFloat(player.style.left) / 100) * COLS - 0.5)
  const row = Math.round((parseFloat(player.style.top) / 100) * ROWS - 0.5)

  return { row, col }
}

// Find the cheese's current grid position.
function getCheesePosition() {
  const cheese = screen.getByRole('img', {
    name: 'Cheese goal',
  }) as HTMLElement

  return {
    row: Number(cheese.style.gridRow) - 1,
    col: Number(cheese.style.gridColumn) - 1,
  }
}

// Move Pavlos from his current position to the cheese.
// Our test maze has no internal walls, so a direct route is valid.
function reachCheese() {
  const start = getPlayerPosition()
  const goal = getCheesePosition()

  const horizontalKey = goal.col > start.col ? 'ArrowRight' : 'ArrowLeft'
  const verticalKey = goal.row > start.row ? 'ArrowDown' : 'ArrowUp'

  for (let i = 0; i < Math.abs(goal.col - start.col); i++) {
    pressArrow(horizontalKey)
  }

  for (let i = 0; i < Math.abs(goal.row - start.row); i++) {
    pressArrow(verticalKey)
  }
}

beforeEach(() => {
  // Reset the mock before each test.
  mockedGenerateMaze.mockReset()
})

describe('Maze', () => {
  it('renders the maze and cheese', () => {
    setupMaze()

    expect(
      screen.getByRole('group', { name: 'Maze' }),
    ).toBeInTheDocument()

    expect(
      screen.getByRole('img', { name: 'Cheese goal' }),
    ).toBeInTheDocument()

    expect(document.querySelectorAll('.maze-cell')).toHaveLength(ROWS * COLS)
  })

  it('moves Pavlos using arrow keys', () => {
    setupMaze()

    const start = getPlayerPosition()

    // Move toward the center from whichever corner Pavlos starts in.
    const key = start.col === 0 ? 'ArrowRight' : 'ArrowLeft'

    pressArrow(key)

    const current = getPlayerPosition()

    expect(current.row).toBe(start.row)
    expect(current.col).toBe(
      start.col === 0 ? start.col + 1 : start.col - 1,
    )
  })

  it('prevents Pavlos from moving through walls', () => {
    setupMaze()

    const start = getPlayerPosition()

    // Try to move outside the maze from the current corner.
    const horizontalKey = start.col === 0 ? 'ArrowLeft' : 'ArrowRight'
    const verticalKey = start.row === 0 ? 'ArrowUp' : 'ArrowDown'

    pressArrow(horizontalKey)
    pressArrow(verticalKey)

    expect(getPlayerPosition()).toEqual(start)
  })

  it('shows the victory message after reaching the cheese', () => {
    setupMaze()

    // The button should not exist before winning.
    expect(
      screen.queryByRole('button', { name: 'Try Another Maze' }),
    ).not.toBeInTheDocument()

    const goal = getCheesePosition()

    reachCheese()

    expect(getPlayerPosition()).toEqual(goal)

    expect(
      screen.getByText(/You found the cheese!/),
    ).toBeInTheDocument()

    expect(
      screen.getByRole('button', { name: 'Try Another Maze' }),
    ).toBeInTheDocument()

    // The cheese disappears after winning.
    expect(
      screen.queryByRole('img', { name: 'Cheese goal' }),
    ).not.toBeInTheDocument()
  })

  it('stops Pavlos from moving after winning', () => {
    setupMaze()

    reachCheese()

    const winningPosition = getPlayerPosition()

    pressArrow('ArrowUp')
    pressArrow('ArrowDown')
    pressArrow('ArrowLeft')
    pressArrow('ArrowRight')

    expect(getPlayerPosition()).toEqual(winningPosition)
  })

  it('generates another maze and resets Pavlos after clicking restart', () => {
    setupMaze()

    // The first maze is generated when the component mounts.
    expect(mockedGenerateMaze).toHaveBeenCalledTimes(1)

    reachCheese()

    const restartButton = screen.getByRole('button', {
      name: 'Try Another Maze',
    })

    fireEvent.click(restartButton)

    // A second maze should be generated.
    expect(mockedGenerateMaze).toHaveBeenCalledTimes(2)

    // The victory message should disappear.
    expect(
      screen.queryByText(/You found the cheese!/),
    ).not.toBeInTheDocument()

    // The cheese should reappear.
    expect(
      screen.getByRole('img', { name: 'Cheese goal' }),
    ).toBeInTheDocument()

    // Pavlos should restart at a valid corner.
    const start = getPlayerPosition()
    const validRows = [0, ROWS - 1]
    const validCols = [0, COLS - 1]

    expect(validRows).toContain(start.row)
    expect(validCols).toContain(start.col)

    // Pavlos should display his starting smile.
    const playerImage = screen.getByAltText('Pavlos player')
    expect(playerImage).toHaveAttribute('src', '/pavlos_smile.png')
  })
})

describe('Maze gaze steering', () => {
  // A 900 px tall viewport with 60 px cells: the drive ellipse reaches
  // 0.25 * 900 = 225 px (3.75 cells) sideways, the pull ring 1.5x that.
  const CELL = 60

  beforeEach(() => {
    vi.useFakeTimers()
    vi.stubGlobal('innerWidth', 900)
    vi.stubGlobal('innerHeight', 900)
    vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockReturnValue(
      { left: 0, top: 0, width: COLS * CELL, height: ROWS * CELL, right: COLS * CELL, bottom: ROWS * CELL, x: 0, y: 0, toJSON: () => ({}) },
    )
    mockedGenerateMaze.mockImplementation(createTestMaze)
  })

  afterEach(() => {
    vi.useRealTimers()
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
  })

  // Gaze at the centre of a cell, as viewport fractions.
  const gazeAtCell = (row: number, col: number) => ({ x: ((col + 0.5) * CELL) / 900, y: ((row + 0.5) * CELL) / 900 })

  // Two cells from the start, toward the middle of the maze.
  function inwardTarget(start: { row: number; col: number }, cells: number) {
    return { row: start.row, col: start.col === 0 ? start.col + cells : start.col - cells }
  }

  it('drives Pavlos to the cell he is looking at, one step per tick, then stops there', () => {
    const { rerender } = render(<Maze />)
    const start = getPlayerPosition()
    const target = inwardTarget(start, 2)

    rerender(<Maze gazePoint={gazeAtCell(target.row, target.col)} />)
    expect(getPlayerPosition()).toEqual(inwardTarget(start, 1)) // first step is immediate

    act(() => { vi.advanceTimersByTime(160) })
    expect(getPlayerPosition()).toEqual(target)

    act(() => { vi.advanceTimersByTime(1000) })
    expect(getPlayerPosition()).toEqual(target)
  })

  it('ignores a glance just outside the drive ellipse, but a held stare pulls Pavlos there', () => {
    const { rerender } = render(<Maze />)
    const start = getPlayerPosition()
    const near = inwardTarget(start, 5) // 300 px: past the ellipse (225 px), inside the pull ring (337 px)

    rerender(<Maze gazePoint={gazeAtCell(near.row, near.col)} />)
    act(() => { vi.advanceTimersByTime(GAZE_DWELL_MS - 100) })
    expect(getPlayerPosition()).toEqual(start)

    act(() => { vi.advanceTimersByTime(2000) })
    expect(getPlayerPosition()).toEqual(near)
  })

  it('ignores even a held stare beyond the pull ring (looking around)', () => {
    const { rerender } = render(<Maze />)
    const start = getPlayerPosition()
    const far = inwardTarget(start, 8)

    rerender(<Maze gazePoint={gazeAtCell(far.row, far.col)} />)
    act(() => { vi.advanceTimersByTime(3000) })
    expect(getPlayerPosition()).toEqual(start)
  })

  it('stops when gaze is lost', () => {
    const { rerender } = render(<Maze />)
    const start = getPlayerPosition()
    const target = inwardTarget(start, 2)

    rerender(<Maze gazePoint={gazeAtCell(target.row, target.col)} />)
    rerender(<Maze gazePoint={null} />)
    act(() => { vi.advanceTimersByTime(1000) })
    expect(getPlayerPosition()).toEqual(inwardTarget(start, 1))
  })

  it('follows a moving gaze point from wherever Pavlos is now', () => {
    const { rerender } = render(<Maze />)
    const start = getPlayerPosition()
    const one = inwardTarget(start, 1)
    const three = inwardTarget(start, 3)

    rerender(<Maze gazePoint={gazeAtCell(one.row, one.col)} />)
    act(() => { vi.advanceTimersByTime(500) })
    expect(getPlayerPosition()).toEqual(one)

    // Three cells from the start is only two from Pavlos now: inside the ellipse.
    rerender(<Maze gazePoint={gazeAtCell(three.row, three.col)} />)
    act(() => { vi.advanceTimersByTime(500) })
    expect(getPlayerPosition()).toEqual(three)
  })

  it('outlines the drive area around Pavlos only while gaze is steering', () => {
    const { rerender } = render(<Maze />)
    expect(screen.queryByTestId('drive-area')).not.toBeInTheDocument()

    const start = getPlayerPosition()
    const target = inwardTarget(start, 1)
    rerender(<Maze gazePoint={gazeAtCell(target.row, target.col)} />)
    const area = screen.getByTestId('drive-area')
    const player = document.querySelector('.maze-player') as HTMLElement
    expect(area.style.left).toBe(player.style.left)
    expect(area.style.top).toBe(player.style.top)
    expect(area.style.getPropertyValue('--drive-rx')).toBe(String(GAZE_DRIVE_RADIUS.x))
    expect(area.style.getPropertyValue('--drive-ry')).toBe(String(GAZE_DRIVE_RADIUS.y))
    const pull = screen.getByTestId('pull-area')
    expect(pull.style.left).toBe(player.style.left)
    expect(pull.style.getPropertyValue('--drive-rx')).toBe(String(GAZE_DRIVE_RADIUS.x * GAZE_PULL_SCALE))

    rerender(<Maze gazePoint={null} />)
    expect(screen.queryByTestId('drive-area')).not.toBeInTheDocument()
    expect(screen.queryByTestId('pull-area')).not.toBeInTheDocument()
  })

  describe('recentering', () => {
    const recenterPhase = () => screen.getByRole('group', { name: 'Maze' }).dataset.recenter

    /** Keep predicting `point` (a fresh object each time, like the gaze loop) for `ms`. */
    function stare(rerender: (ui: React.ReactElement) => void, point: { x: number; y: number }, ms: number, onRecenter = vi.fn()) {
      for (let t = 0; t < ms; t += 200) {
        rerender(<Maze gazePoint={{ ...point }} onRecenter={onRecenter} />)
        act(() => { vi.advanceTimersByTime(200) })
      }
    }

    it('R starts a hands-free recenter: Pavlos holds still, and the stare gives the shift', () => {
      const onRecenter = vi.fn()
      const { rerender } = render(<Maze />)
      const start = getPlayerPosition()
      rerender(<Maze gazePoint={gazeAtCell(start.row, start.col)} onRecenter={onRecenter} />)
      pressKey('r')
      expect(recenterPhase()).toBe('confirming')

      // The user stares at Pavlos, but gaze reads one cell inward: normally that drives him.
      const reads = inwardTarget(start, 1)
      // Samples arrive every 200 ms: the ninth, 1.6 s into the stare, confirms it.
      stare(rerender, gazeAtCell(reads.row, reads.col), 1800, onRecenter)
      expect(recenterPhase()).toBe('capturing')
      stare(rerender, gazeAtCell(reads.row, reads.col), 1000, onRecenter)
      expect(getPlayerPosition()).toEqual(start)

      expect(onRecenter).toHaveBeenCalledOnce()
      const [shift] = onRecenter.mock.calls[0]
      const pavlos = gazeAtCell(start.row, start.col)
      const seen = gazeAtCell(reads.row, reads.col)
      expect(shift.x).toBeCloseTo(pavlos.x - seen.x, 12)
      expect(shift.y).toBeCloseTo(0, 12)

      // He keeps still while the result shows, then play resumes.
      expect(recenterPhase()).toBe('done')
      act(() => { vi.advanceTimersByTime(RECENTER_RESULT_MS - 100) })
      expect(getPlayerPosition()).toEqual(start)
      act(() => { vi.advanceTimersByTime(100) })
      expect(recenterPhase()).toBe('idle')
    })

    it('R again cancels, and losing gaze cancels', () => {
      const { rerender } = render(<Maze />)
      const start = getPlayerPosition()
      rerender(<Maze gazePoint={gazeAtCell(start.row, start.col)} />)
      pressKey('r')
      pressKey('r')
      expect(recenterPhase()).toBe('idle')

      pressKey('r')
      rerender(<Maze gazePoint={null} />)
      expect(recenterPhase()).toBe('idle')
    })

    it('needs gaze to start', () => {
      render(<Maze />)
      pressKey('r')
      expect(recenterPhase()).toBe('idle')
    })
  })
})

// Press and release any key.
function pressKey(key: string) {
  act(() => {
    fireEvent.keyDown(window, { key })
    fireEvent.keyUp(window, { key })
  })
}
