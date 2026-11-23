import { useCallback, useEffect, useRef, useState, type CSSProperties } from 'react'
import { generateMaze } from './mazeGenerator'
import type { Maze as MazeGrid, MazePosition } from './mazeGenerator'
import { decideDrive, gazeRelation, type ScreenRect } from './driveAssist'
import { isDwelling, trackFixation, type Fixation } from './dwell'
import { gazeInZone, NO_ZONE_DWELL, trackZoneDwell, zoneProgress, type ZoneDwell } from './gazeZone'
import { RecenterCue } from './RecenterCue'
import { RecenterStrip } from './RecenterStrip'
import { isStuckPrompt, NEW_STUCK, quietStuck, trackStuck, type StuckState } from './stuck'
import {
  holdsPavlos,
  isRecentering,
  recenterProgress,
  sampleRecenter,
  stareHeld,
  startRecenter,
  type GazeShift,
  type RecenterState,
} from './recenter'
import type { Direction } from './gazeDirection'
import type { ScreenFraction } from '../vision/model'
import { currentViewport, type Viewport } from '../vision/viewport'
import {
  GAZE_DRIVE_RADIUS,
  GAZE_DWELL_MS,
  GAZE_DWELL_RADIUS,
  GAZE_MOVE_INTERVAL_MS,
  GAZE_PULL_SCALE,
  GAZE_TARGET_HOLD_CELLS,
  KEYBOARD_MOVE_INTERVAL_MS,
  MAZE_COLS,
  MAZE_ROWS,
  PLAYER_SIZE_PERCENT,
  RECENTER,
  RECENTER_RESULT_MS,
  RECENTER_STRIP,
  SHOW_DRIVE_AREA,
  STUCK,
} from '../config'
import './Maze.css'

const ROWS = MAZE_ROWS
const COLS = MAZE_COLS

// Hide Pavlos's text once a stare has lasted this many predictions (about 0.4 s).
const RECENTER_BUBBLE_HIDE_SAMPLES = 3

interface Game {
  maze: MazeGrid
  cheese: MazePosition
  player: MazePosition & { rotation: number; hasMoved: boolean }
}

// Move Pavlos one cell, unless a wall, the maze edge, or a win stops him.
function stepPlayer(game: Game, direction: Direction): Game {
  const current = game.player

  // Stop movement once Pavlos reaches the cheese.
  if (current.row === game.cheese.row && current.col === game.cheese.col) {
    return game
  }

  const cell = game.maze[current.row][current.col]
  let nextRow = current.row
  let nextCol = current.col

  // Check for a wall before moving.
  switch (direction) {
    case 'up':
      if (cell.walls.top) return game
      nextRow -= 1
      break
    case 'down':
      if (cell.walls.bottom) return game
      nextRow += 1
      break
    case 'left':
      if (cell.walls.left) return game
      nextCol -= 1
      break
    case 'right':
      if (cell.walls.right) return game
      nextCol += 1
      break
  }

  // Keep the player inside the maze.
  if (nextRow < 0 || nextRow >= ROWS || nextCol < 0 || nextCol >= COLS) {
    return game
  }

  return {
    ...game,
    player: {
      row: nextRow,
      col: nextCol,
      rotation:
        current.rotation +
        (direction === 'right' || direction === 'down' ? 90 : -90),
      hasMoved: true,
    },
  }
}

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
  /** Latest smoothed gaze, as viewport fractions; null while gaze is unavailable. */
  gazePoint?: ScreenFraction | null
  /** A recenter finished: add `shift` to future gaze points. */
  onRecenter?: (shift: GazeShift) => void
}

// Pavlos's centre on screen, as viewport fractions.
function pavlosCentre(rect: ScreenRect, player: MazePosition, viewport: Viewport): ScreenFraction {
  return {
    x: (rect.left + ((player.col + 0.5) * rect.width) / COLS) / viewport.width,
    y: (rect.top + ((player.row + 0.5) * rect.height) / ROWS) / viewport.height,
  }
}

export default function Maze({ gazePoint = null, onRecenter }: MazeProps) {
  const mazeRef = useRef<HTMLDivElement>(null)

  // Generate the maze and starting positions together.
  const [game, setGame] = useState<Game>(() => {
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
      setGame((currentGame) => stepPlayer(currentGame, direction))
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
      timer = setInterval(() => movePlayer(direction), KEYBOARD_MOVE_INTERVAL_MS)
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


  // Steer by gaze. Each tick re-decides from Pavlos's current cell and the
  // latest gaze point, so he stops on arriving instead of running on until
  // the next prediction.
  const gazeRef = useRef(gazePoint)
  const fixationRef = useRef<Fixation | null>(null)
  const gameRef = useRef(game)
  useEffect(() => {
    gameRef.current = game
  }, [game])

  // Dynamic recentering: a stare at Pavlos measures and removes a constant
  // gaze offset. He holds still while it runs.
  const [recenter, setRecenter] = useState<RecenterState>({ phase: 'idle' })
  const holdRef = useRef(false)
  useEffect(() => {
    holdRef.current = holdsPavlos(recenter)
  }, [recenter])

  // The Recenter strip: looking at it for RECENTER_STRIP.dwellMs starts recentering.
  const columnRef = useRef<HTMLDivElement>(null)
  const stripRef = useRef<HTMLButtonElement>(null)
  const stripDwellRef = useRef<ZoneDwell>(NO_ZONE_DWELL)
  const [stripProgress, setStripProgress] = useState(0)
  // Time of the latest gaze sample, for drawing stare progress.
  const [sampleTime, setSampleTime] = useState(0)
  // Stuck detection, which only suggests recentering.
  const stuckRef = useRef<StuckState>(NEW_STUCK)
  const [stuckPrompt, setStuckPrompt] = useState(false)

  useEffect(() => {
    gazeRef.current = gazePoint
    const now = Date.now()
    const viewport = currentViewport()
    fixationRef.current = trackFixation(fixationRef.current, gazePoint, now, viewport, GAZE_DWELL_RADIUS)
    setSampleTime(now)

    if (gazePoint === null) {
      stripDwellRef.current = NO_ZONE_DWELL
      setStripProgress(0)
      stuckRef.current = NEW_STUCK
      setStuckPrompt(false)
      setRecenter((state) => (isRecentering(state) ? { phase: 'idle' } : state))
      return
    }

    const rect = mazeRef.current?.getBoundingClientRect()
    if (rect) {
      const target = pavlosCentre(rect, gameRef.current.player, viewport)
      setRecenter((state) => sampleRecenter(state, { point: gazePoint, now, target, viewport }, RECENTER))
    }

    // Stuck: Pavlos stays put while the user steers near him, or rocks between two cells.
    const current = gameRef.current
    const won = current.player.row === current.cheese.row && current.player.col === current.cheese.col
    if (holdRef.current || won) {
      stuckRef.current = quietStuck(stuckRef.current, now, STUCK)
    } else if (rect) {
      const relation = gazeRelation(gazePoint, current.player, current.maze, rect, viewport, GAZE_DRIVE_RADIUS)
      const trying = !!relation && relation.reach <= GAZE_PULL_SCALE ** 2 && !relation.inOwnCell
      stuckRef.current = trackStuck(stuckRef.current, { now, player: current.player, trying }, STUCK)
    }
    setStuckPrompt(isStuckPrompt(stuckRef.current))

    const column = columnRef.current?.getBoundingClientRect()
    const strip = stripRef.current?.getBoundingClientRect()
    if (!column || !strip || holdRef.current) {
      stripDwellRef.current = NO_ZONE_DWELL
      setStripProgress(0)
      return
    }
    // From the strip's column past the screen edge, and a margin above and below.
    const pad = RECENTER_STRIP.pad * viewport.height
    const zone = { left: column.left, top: strip.top - pad, width: Infinity, height: strip.height + 2 * pad }
    stripDwellRef.current = trackZoneDwell(
      stripDwellRef.current, gazeInZone(gazePoint, zone, viewport), now, RECENTER_STRIP.maxMisses,
    )
    const progress = zoneProgress(stripDwellRef.current, now, RECENTER_STRIP.dwellMs)
    if (progress < 1) {
      setStripProgress(progress)
      return
    }
    stripDwellRef.current = NO_ZONE_DWELL
    setStripProgress(0)
    setRecenter(startRecenter(now))
  }, [gazePoint])

  // Hand a finished shift to the gaze pipeline (once per result), show the
  // result briefly, then resume.
  const onRecenterRef = useRef(onRecenter)
  useEffect(() => {
    onRecenterRef.current = onRecenter
  }, [onRecenter])
  useEffect(() => {
    if (recenter.phase === 'done') onRecenterRef.current?.(recenter.shift)
    if (recenter.phase !== 'done' && recenter.phase !== 'gaveUp') return
    const timer = setTimeout(() => setRecenter({ phase: 'idle' }), RECENTER_RESULT_MS)
    return () => clearTimeout(timer)
  }, [recenter])

  // Start recentering, or cancel it if it is running. Needs gaze.
  const toggleRecenter = useCallback(() => {
    if (gazeRef.current === null) return
    setRecenter((state) => (isRecentering(state) ? { phase: 'idle' } : startRecenter(Date.now())))
  }, [])

  // R does the same as clicking the Recenter strip.
  useEffect(() => {
    function handleKeyDown(event: KeyboardEvent) {
      if (event.key !== 'r' && event.key !== 'R') return
      if (event.repeat || event.ctrlKey || event.metaKey || event.altKey) return
      toggleRecenter()
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [toggleRecenter])

  const hasGaze = gazePoint !== null
  useEffect(() => {
    if (!hasGaze) return
    // The cell Pavlos is heading for (or resting in), kept against jitter.
    let heldTarget: MazePosition | null = null

    function tick() {
      if (holdRef.current) return
      const rect = mazeRef.current?.getBoundingClientRect()
      if (!rect) return
      const viewport = currentViewport()
      const dwelling = isDwelling(fixationRef.current, Date.now(), GAZE_DWELL_MS)
      setGame((currentGame) => {
        const decision = decideDrive(
          gazeRef.current, currentGame.player, currentGame.maze, rect, viewport,
          {
            radius: GAZE_DRIVE_RADIUS,
            pullScale: GAZE_PULL_SCALE,
            dwelling,
            heldTarget,
            holdCells: GAZE_TARGET_HOLD_CELLS,
          },
        )
        // Idempotent, so a repeated updater call (React strict mode) agrees.
        heldTarget = decision.kind === 'drive'
          ? decision.target
          : decision.kind === 'stop' ? currentGame.player : null
        return decision.kind === 'drive' ? stepPlayer(currentGame, decision.path[0]) : currentGame
      })
    }

    tick()
    const timer = setInterval(tick, GAZE_MOVE_INTERVAL_MS)
    return () => clearInterval(timer)
  }, [hasGaze])

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
        ref={mazeRef}
        className="maze"
        role="group"
        aria-label="Maze"
        data-recenter={recenter.phase}
        style={{
          gridTemplateColumns: `repeat(${COLS}, minmax(0, 1fr))`,
          gridTemplateRows: `repeat(${ROWS}, minmax(0, 1fr))`,
          // Square cells, whatever the grid shape.
          aspectRatio: `${COLS} / ${ROWS}`,
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

        {/* Outline where gaze steers Pavlos, and the ring where a stare pulls him; both move with him */}
        {SHOW_DRIVE_AREA && hasGaze && !hasWon && (
          <>
            <div
              className="maze-drive-area maze-pull-area"
              data-testid="pull-area"
              aria-hidden="true"
              style={{
                left: `${((player.col + 0.5) / COLS) * 100}%`,
                top: `${((player.row + 0.5) / ROWS) * 100}%`,
                '--drive-rx': GAZE_DRIVE_RADIUS.x * GAZE_PULL_SCALE,
                '--drive-ry': GAZE_DRIVE_RADIUS.y * GAZE_PULL_SCALE,
              } as CSSProperties}
            />
            <div
              className="maze-drive-area"
              data-testid="drive-area"
              aria-hidden="true"
              style={{
                left: `${((player.col + 0.5) / COLS) * 100}%`,
                top: `${((player.row + 0.5) / ROWS) * 100}%`,
                '--drive-rx': GAZE_DRIVE_RADIUS.x,
                '--drive-ry': GAZE_DRIVE_RADIUS.y,
              } as CSSProperties}
            />
          </>
        )}

        {/* Display Pavlos at his current position */}
        <div
          className="maze-player"
          style={{
            left: `${((player.col + 0.5) / COLS) * 100}%`,
            top: `${((player.row + 0.5) / ROWS) * 100}%`,
            width: `${PLAYER_SIZE_PERCENT / COLS}%`,
            height: `${PLAYER_SIZE_PERCENT / ROWS}%`,
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

        {/* What Pavlos says while recentering, and the stare progress ring */}
        <RecenterCue
          state={recenter}
          progress={recenterProgress(recenter, sampleTime, RECENTER)}
          held={stareHeld(recenter, RECENTER_BUBBLE_HIDE_SAMPLES)}
          stuck={stuckPrompt}
          player={player}
          rows={ROWS}
          cols={COLS}
        />
      </div>

      {/* The Recenter strip takes the right-hand column until the win message does */}
      {hasGaze && !hasWon && (
        <RecenterStrip
          state={recenter}
          progress={stripProgress}
          prompt={stuckPrompt}
          onClick={toggleRecenter}
          columnRef={columnRef}
          stripRef={stripRef}
        />
      )}

      {/* Show the victory message and restart button after winning */}
      {hasWon && (
        <div className="maze-win">
          <div className="maze-win-trophy" aria-hidden="true">🏆</div>
          <h2>You found the cheese!</h2>
          <p>Pavlos found the cheese!</p>
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
