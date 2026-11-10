import { useState } from 'react'
import { PLAYER_SIZE_PERCENT } from '../config'
import type { Cell, RaceMazeData } from './raceClient'
import '../maze/Maze.css'
import './race.css'

interface Props {
  maze: RaceMazeData
  me: Cell
  /** The opponent, drawn as a faded ghost in the same maze. Null hides it. */
  ghost: Cell | null
  showCheese: boolean
  /** Which players have won, so they stop rolling and smile. */
  meWon?: boolean
  ghostWon?: boolean
}

const wall = (closed: boolean) => (closed ? '2px solid var(--maze-wall)' : 'none')

function placement(cell: Cell, maze: RaceMazeData) {
  return {
    left: `${((cell.col + 0.5) / maze.cols) * 100}%`,
    top: `${((cell.row + 0.5) / maze.rows) * 100}%`,
    width: `${PLAYER_SIZE_PERCENT / maze.cols}%`,
    height: `${PLAYER_SIZE_PERCENT / maze.rows}%`,
  }
}

/**
 * Pavlos rolls like in the solo maze: a quarter turn per step, clockwise for
 * right/down and counter-clockwise for left/up. He smiles until he first moves
 * and again once he wins.
 */
function RollingPavlos({
  cell,
  maze,
  won,
  ghost,
}: {
  cell: Cell
  maze: RaceMazeData
  won: boolean
  ghost: boolean
}) {
  const [previous, setPrevious] = useState(cell)
  const [rotation, setRotation] = useState(0)
  const [hasMoved, setHasMoved] = useState(false)

  // The server sent a new position: roll one quarter turn in that direction.
  if (previous.row !== cell.row || previous.col !== cell.col) {
    const forward = cell.row > previous.row || cell.col > previous.col
    setPrevious(cell)
    setRotation(rotation + (forward ? 90 : -90))
    setHasMoved(true)
  }

  return (
    <div
      className={ghost ? 'maze-player race-ghost' : 'maze-player'}
      data-testid={ghost ? 'race-ghost' : 'race-me'}
      style={placement(cell, maze)}
    >
      <img
        src={!hasMoved || won ? '/pavlos_smile.png' : '/pavlos_frown.png'}
        alt={ghost ? 'Opponent' : 'You'}
        draggable={false}
        style={{ transform: `rotate(${won ? 0 : rotation}deg)` }}
      />
    </div>
  )
}

/** Draws the server's maze, the player, and the opponent's ghost. */
export function RaceMaze({ maze, me, ghost, showCheese, meWon = false, ghostWon = false }: Props) {
  return (
    <div
      className="maze"
      role="group"
      aria-label="Race maze"
      style={{
        gridTemplateColumns: `repeat(${maze.cols}, minmax(0, 1fr))`,
        gridTemplateRows: `repeat(${maze.rows}, minmax(0, 1fr))`,
      }}
    >
      {maze.walls.map((row, r) =>
        row.map((walls, c) => (
          <div
            key={`${r}-${c}`}
            className="maze-cell"
            style={{
              gridRow: r + 1,
              gridColumn: c + 1,
              borderTop: wall(walls.top),
              borderRight: wall(walls.right),
              borderBottom: wall(walls.bottom),
              borderLeft: wall(walls.left),
            }}
          />
        )),
      )}

      {showCheese && (
        <div
          className="maze-cheese"
          role="img"
          aria-label="Cheese goal"
          style={{ gridRow: maze.cheese.row + 1, gridColumn: maze.cheese.col + 1 }}
        >
          <img src="/cheese.svg" alt="" draggable={false} />
        </div>
      )}

      {ghost && <RollingPavlos cell={ghost} maze={maze} won={ghostWon} ghost />}
      <RollingPavlos cell={me} maze={maze} won={meWon} ghost={false} />
    </div>
  )
}
