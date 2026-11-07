import { useCallback, useEffect, useState } from 'react'
import { generateMaze } from './mazeGenerator'
import type { MazePosition } from './mazeGenerator'
import './Maze.css'

const ROWS = 15
const COLS = 15
const MOVE_INTERVAL = 160

type Direction = 'up' | 'down' | 'left' | 'right'

const KEY_DIRECTIONS: Record<string, Direction> = {
  ArrowUp: 'up',
  ArrowDown: 'down',
  ArrowLeft: 'left',
  ArrowRight: 'right',
}

// Randomly select two corners on opposite horizontal sides.
// Pavlos can start on either the left or right.
// Both characters can independently start at the top or bottom.
function randomPositions(): {
  player: MazePosition
  cheese: MazePosition
} {
  const playerOnLeft = Math.random() < 0.5
  const playerOnTop = Math.random() < 0.5
  const cheeseOnTop = Math.random() < 0.5

  const player: MazePosition = {
    row: playerOnTop ? 0 : ROWS - 1,
    col: playerOnLeft ? 0 : COLS - 1,
  }

  const cheese: MazePosition = {
    row: cheeseOnTop ? 0 : ROWS - 1,
    col: playerOnLeft ? COLS - 1 : 0,
  }

  return { player, cheese }
}

interface MazeProps {
  gazeDirection?: 'up' | 'down' | 'left' | 'right' | null
}

export default function Maze({ gazeDirection = null }: MazeProps) {
  // Generate the maze and starting positions together.
  const [game, setGame] = useState(() => {
    const positions = randomPositions()

    return {
      maze: generateMaze(
        ROWS,
        COLS,
        positions.player,
        positions.cheese,
      ),
      cheese: positions.cheese,
      player: {
        ...positions.player,
        rotation: 0,
        hasMoved: false,
      },
    }
  })

  const { maze, cheese, player } = game

  const movePlayer = useCallback(
    (direction: Direction) => {
      setGame((currentGame) => {
        const current = currentGame.player

        // Stop movement once Pavlos reaches the cheese.
        if (
          current.row === currentGame.cheese.row &&
          current.col === currentGame.cheese.col
        ) {
          return currentGame
        }

        const cell = currentGame.maze[current.row][current.col]
        let nextRow = current.row
        let nextCol = current.col

        // Check for a wall before moving.
        switch (direction) {
          case 'up':
            if (cell.walls.top) return currentGame
            nextRow -= 1
            break
          case 'down':
            if (cell.walls.bottom) return currentGame
            nextRow += 1
            break
          case 'left':
            if (cell.walls.left) return currentGame
            nextCol -= 1
            break
          case 'right':
            if (cell.walls.right) return currentGame
            nextCol += 1
            break
        }

        // Keep the player inside the maze.
        if (
          nextRow < 0 ||
          nextRow >= ROWS ||
          nextCol < 0 ||
          nextCol >= COLS
        ) {
          return currentGame
        }

        return {
          ...currentGame,
          player: {
            row: nextRow,
            col: nextCol,
            rotation:
              current.rotation +
              (direction === 'right' || direction === 'down' ? 90 : -90),
            hasMoved: true,
          },
        }
      })
    },
    [],
  )

  // Move repeatedly while an arrow key is held.
  useEffect(() => {
    let timer: ReturnType<typeof setInterval> | undefined
    let activeKey: string | null = null

    function stopMoving() {
      clearInterval(timer)
      timer = undefined
      activeKey = null
    }

    function handleKeyDown(event: KeyboardEvent) {
      const direction = KEY_DIRECTIONS[event.key]
      if (!direction) return

      event.preventDefault()

      // Our timer handles repeated movement.
      if (event.repeat || activeKey === event.key) return

      stopMoving()
      activeKey = event.key

      movePlayer(direction)
      timer = setInterval(() => movePlayer(direction), MOVE_INTERVAL)
    }

    function handleKeyUp(event: KeyboardEvent) {
      if (event.key === activeKey) stopMoving()
    }

    function handleVisibilityChange() {
      if (document.hidden) stopMoving()
    }

    window.addEventListener('keydown', handleKeyDown)
    window.addEventListener('keyup', handleKeyUp)
    window.addEventListener('blur', stopMoving)
    document.addEventListener('visibilitychange', handleVisibilityChange)

    return () => {
      stopMoving()
      window.removeEventListener('keydown', handleKeyDown)
      window.removeEventListener('keyup', handleKeyUp)
      window.removeEventListener('blur', stopMoving)
      document.removeEventListener('visibilitychange', handleVisibilityChange)
    }
  }, [movePlayer])


  // Move Pavlos using gaze predictions.
useEffect(() => {
  if (!gazeDirection) return

  movePlayer(gazeDirection)

  const timer = setInterval(() => {
    movePlayer(gazeDirection)
  }, MOVE_INTERVAL)

  return () => clearInterval(timer)
}, [gazeDirection, movePlayer])

  // Winning depends on the cheese's randomized position.
  const hasWon = player.row === cheese.row && player.col === cheese.col

  function restartMaze() {
    const positions = randomPositions()

    // Generate another maze with new starting and ending positions.
    setGame({
      maze: generateMaze(
        ROWS,
        COLS,
        positions.player,
        positions.cheese,
      ),
      cheese: positions.cheese,
      player: {
        ...positions.player,
        rotation: 0,
        hasMoved: false,
      },
    })
  }

  return (
    <>
      {/* Display the generated maze */}
      <div
        className="maze"
        role="group"
        aria-label="Maze"
        style={{
          gridTemplateColumns: `repeat(${COLS}, minmax(0, 1fr))`,
          gridTemplateRows: `repeat(${ROWS}, minmax(0, 1fr))`,
        }}
      >
        {/* Render each cell and its walls */}
        {maze.map((row) =>
          row.map((cell) => (
            <div
              key={`${cell.row}-${cell.col}`}
              className="maze-cell"
              style={{
                gridRow: cell.row + 1,
                gridColumn: cell.col + 1,
                borderTop: cell.walls.top
                  ? '2px solid var(--maze-wall)'
                  : 'none',
                borderRight: cell.walls.right
                  ? '2px solid var(--maze-wall)'
                  : 'none',
                borderBottom: cell.walls.bottom
                  ? '2px solid var(--maze-wall)'
                  : 'none',
                borderLeft: cell.walls.left
                  ? '2px solid var(--maze-wall)'
                  : 'none',
              }}
            />
          )),
        )}

        {/* Display the cheese until Pavlos reaches it */}
        {!hasWon && (
          <div
            className="maze-cheese"
            role="img"
            aria-label="Cheese goal"
            style={{
              gridRow: cheese.row + 1,
              gridColumn: cheese.col + 1,
            }}
          >
            <img src="/cheese.svg" alt="" draggable={false} />
          </div>
        )}

        {/* Display Pavlos at his current position */}
        <div
          className="maze-player"
          style={{
            left: `${((player.col + 0.5) / COLS) * 100}%`,
            top: `${((player.row + 0.5) / ROWS) * 100}%`,
            width: `${75 / COLS}%`,
            height: `${75 / ROWS}%`,
          }}
        >
          <img
            src={
              !player.hasMoved || hasWon
                ? '/pavlos_smile.png'
                : '/pavlos_frown.png'
            }
            alt="Pavlos player"
            draggable={false}
            style={{
              transform: `rotate(${hasWon ? 0 : player.rotation}deg)`,
            }}
          />
        </div>
      </div>

      {/* Show the victory message and restart button after winning */}
      {hasWon && (
        <div className="maze-win">
          <h2>🎉 You found the cheese!</h2>
          <button
            type="button"
            className="game-button"
            onClick={restartMaze}
          >
            Try Another Maze
          </button>
        </div>
      )}
    </>
  )
}
