import { beforeEach, describe, expect, it, vi } from 'vitest'
import { fireEvent, render, screen } from '@testing-library/react'
import Maze from './Maze'
import { generateMaze } from './mazeGenerator'

// Mock the maze generator to create a predictable maze for testing.
vi.mock('./mazeGenerator', () => ({
  generateMaze: vi.fn(),
}))

const mockedGenerateMaze = vi.mocked(generateMaze)

const ROWS = 15
const COLS = 15

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

    expect(document.querySelectorAll('.maze-cell')).toHaveLength(225)
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
