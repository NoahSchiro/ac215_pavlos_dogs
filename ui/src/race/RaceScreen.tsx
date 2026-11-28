import { useCallback, useEffect, useRef, useState } from 'react'
import { GAZE_MOVE_INTERVAL_MS, KEYBOARD_MOVE_INTERVAL_MS, RACE_SERVER_URL, loadRuntimeConfig } from '../config'
import { GazeCrosshair } from '../gaze/GazeCrosshair'
import { useGazeTracking } from '../gaze/useGazeTracking'
import type { Direction } from '../maze/gazeDirection'
import type { CalibrationSession } from '../calibration/record'
import { enterFullscreen } from '../calibration/fullscreen'
import { RaceMaze } from './RaceMaze'
import {
  createRoom,
  moveMessage,
  roomSocketUrl,
  type Cell,
  type RaceMazeData,
  type ServerMessage,
} from './raceClient'
import './race.css'

type Phase = 'lobby' | 'connecting' | 'waiting' | 'racing' | 'finished' | 'opponent_left'

const KEY_DIRECTIONS: Record<string, Direction> = {
  ArrowUp: 'up',
  ArrowDown: 'down',
  ArrowLeft: 'left',
  ArrowRight: 'right',
}

const ERROR_TEXT: Record<string, string> = {
  room_not_found: 'No race with that code. Check the code and try again.',
  room_full: 'That race already has two players.',
  race_started: 'That race has already started.',
}

const LOCAL_HOSTS = new Set(['localhost', '127.0.0.1', '[::1]', '::1'])

/** True when a deployed page would try to reach a race server on localhost. */
function raceServerNotConfigured(serverUrl: string): boolean {
  try {
    return LOCAL_HOSTS.has(new URL(serverUrl).hostname) && !LOCAL_HOSTS.has(window.location.hostname)
  } catch {
    return true
  }
}

/** Phases where the camera and gaze model run, so they are ready when the race starts. */
const TRACKING_PHASES: readonly Phase[] = ['lobby', 'connecting', 'waiting', 'racing']

interface Props {
  onBack: () => void
  /** Race backend address; by default it comes from /config.json. */
  serverUrl?: string
  /** The player's saved calibration, applied the same way as in the solo maze. */
  calibration?: CalibrationSession | null
}

/** Two-player race: create or join a room, then race the opponent's ghost to the cheese. */
export function RaceScreen({ onBack, serverUrl: serverUrlProp, calibration }: Props) {
  // The deployed site names its race server in /config.json; locally it falls
  // back to RACE_SERVER_URL (localhost:8000).
  const [configServerUrl, setConfigServerUrl] = useState<string | null>(null)
  useEffect(() => {
    if (serverUrlProp) return
    let live = true
    void loadRuntimeConfig().then((config) => {
      if (live) setConfigServerUrl(config.raceServerUrl)
    })
    return () => { live = false }
  }, [serverUrlProp])
  const serverUrl = serverUrlProp ?? configServerUrl ?? RACE_SERVER_URL

  const socketRef = useRef<WebSocket | null>(null)
  const [phase, setPhase] = useState<Phase>('lobby')
  const [code, setCode] = useState('')
  const [joinCode, setJoinCode] = useState('')
  const [playerId, setPlayerId] = useState<number | null>(null)
  const [maze, setMaze] = useState<RaceMazeData | null>(null)
  const [positions, setPositions] = useState<Record<string, Cell>>({})
  const [winner, setWinner] = useState<number | null>(null)
  const [error, setError] = useState<string | null>(null)

  // Close the connection when leaving the screen. Clearing the ref first tells
  // onclose this was on purpose.
  useEffect(() => () => {
    const socket = socketRef.current
    socketRef.current = null
    socket?.close()
  }, [])

  const handleMessage = useCallback((message: ServerMessage) => {
    switch (message.type) {
      case 'welcome':
        setPlayerId(message.player)
        setMaze(message.maze)
        setPhase('waiting')
        break
      case 'state':
        setPositions(message.players)
        setWinner(message.winner)
        if (message.winner !== null) setPhase('finished')
        else if (message.started) setPhase('racing')
        break
      case 'opponent_left':
        setPhase((current) => (current === 'finished' ? current : 'opponent_left'))
        break
      case 'error':
        setError(ERROR_TEXT[message.reason] ?? `Could not join the race (${message.reason}).`)
        setPhase('lobby')
        break
      case 'rejected':
        // Walls, moving too fast, etc. The player just doesn't move.
        break
    }
  }, [])

  const connect = useCallback(
    (roomCode: string) => {
      const previous = socketRef.current
      socketRef.current = null
      previous?.close()
      setError(null)
      setCode(roomCode.toUpperCase())
      setPhase('connecting')
      const socket = new WebSocket(roomSocketUrl(serverUrl, roomCode))
      socket.onmessage = (event) => handleMessage(JSON.parse(String(event.data)) as ServerMessage)
      socket.onerror = () => {
        setError('Could not reach the race server.')
        setPhase('lobby')
      }
      // The server closed the connection (restart, network drop). Ignore sockets
      // we replaced or closed ourselves.
      socket.onclose = () => {
        if (socketRef.current !== socket) return
        socketRef.current = null
        setPhase((current) => (current === 'finished' ? current : 'lobby'))
        setError((current) => current ?? 'Lost connection to the race server.')
      }
      socketRef.current = socket
    },
    [serverUrl, handleMessage],
  )

  /** Checks before creating or joining; false (with an error shown) if racing can't work here. */
  const readyToRace = useCallback(() => {
    if (raceServerNotConfigured(serverUrl)) {
      setError('Race mode is not set up on this site yet: no race server is configured.')
      return false
    }
    // Race in full screen, like Start game in the solo maze, so the window
    // matches the calibration. Must run inside the click to be allowed.
    void enterFullscreen()
    return true
  }, [serverUrl])

  const startNewRace = useCallback(async () => {
    if (!readyToRace()) return
    try {
      connect(await createRoom(serverUrl))
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not create a race.')
      setPhase('lobby')
    }
  }, [connect, serverUrl, readyToRace])

  const sendMove = useCallback((direction: Direction) => {
    const socket = socketRef.current
    if (socket && socket.readyState === WebSocket.OPEN) socket.send(moveMessage(direction))
  }, [])

  const opponentId = playerId === 1 ? 2 : 1
  const me = maze && playerId !== null ? (positions[String(playerId)] ?? maze.start) : null
  const ghost = positions[String(opponentId)] ?? null

  return (
    <main className="race-screen">
      <h1>Race a friend</h1>

      {phase === 'lobby' && (
        <div className="race-lobby">
          <button type="button" className="game-button" onClick={() => void startNewRace()}>
            Create race
          </button>
          <form
            className="race-join"
            onSubmit={(event) => {
              event.preventDefault()
              if (joinCode.trim() && readyToRace()) connect(joinCode)
            }}
          >
            <input
              aria-label="Race code"
              placeholder="CODE"
              maxLength={4}
              value={joinCode}
              onChange={(event) => setJoinCode(event.target.value)}
            />
            <button type="submit" className="game-button">
              Join
            </button>
          </form>
          {error && (
            <p className="race-error" role="alert">
              {error}
            </p>
          )}
        </div>
      )}

      {phase === 'connecting' && <p>Connecting...</p>}

      {phase === 'waiting' && (
        <div className="race-lobby">
          <p>Share this code with your opponent:</p>
          <p className="race-code" aria-label="Your race code">
            {code}
          </p>
          <p>Waiting for them to join...</p>
        </div>
      )}

      {maze && me && (phase === 'racing' || phase === 'finished' || phase === 'opponent_left') && (
        <>
          <div className="race-legend">
            <span>You: Pavlos</span>
            <span>Opponent: faded ghost</span>
          </div>
          <RaceMaze
            maze={maze}
            me={me}
            ghost={ghost}
            showCheese={phase !== 'finished'}
            meWon={winner !== null && winner === playerId}
            ghostWon={winner !== null && winner === opponentId}
          />
          {phase === 'finished' && (
            <p className="race-result" role="status">
              {winner === playerId ? 'You found the cheese first!' : 'Your opponent got there first.'}
            </p>
          )}
          {phase === 'opponent_left' && (
            <p className="race-result" role="status">
              Your opponent left the race.
            </p>
          )}
        </>
      )}

      {TRACKING_PHASES.includes(phase) && (
        <RaceControls active={phase === 'racing'} onMove={sendMove} calibration={calibration} />
      )}

      <button type="button" className="back-button" onClick={onBack}>
        Back to Home
      </button>
    </main>
  )
}

/**
 * Steering for the race: gaze (run in this browser) plus the arrow keys.
 * Mounted from the lobby on, so the camera and model load before the race;
 * it only sends moves once the race is on (active).
 */
function RaceControls({ active, onMove, calibration }: {
  active: boolean
  onMove: (direction: Direction) => void
  calibration?: CalibrationSession | null
}) {
  const { videoRef, canvasRef, gazeDirection, gazePoint, statusText, layoutMatches } =
    useGazeTracking(calibration)

  // Gaze: keep moving while the player looks toward an edge of the screen.
  useEffect(() => {
    if (!active || !gazeDirection) return
    onMove(gazeDirection)
    const timer = setInterval(() => onMove(gazeDirection), GAZE_MOVE_INTERVAL_MS)
    return () => clearInterval(timer)
  }, [active, gazeDirection, onMove])

  // Arrow keys: move once on press, then repeat while held.
  useEffect(() => {
    if (!active) return
    let timer: ReturnType<typeof setInterval> | undefined
    let activeKey: string | null = null
    const stop = () => {
      clearInterval(timer)
      activeKey = null
    }
    const down = (event: KeyboardEvent) => {
      const direction = KEY_DIRECTIONS[event.key]
      if (!direction) return
      event.preventDefault()
      if (event.repeat || activeKey === event.key) return
      stop()
      activeKey = event.key
      onMove(direction)
      timer = setInterval(() => onMove(direction), KEYBOARD_MOVE_INTERVAL_MS)
    }
    const up = (event: KeyboardEvent) => {
      if (event.key === activeKey) stop()
    }
    window.addEventListener('keydown', down)
    window.addEventListener('keyup', up)
    window.addEventListener('blur', stop)
    return () => {
      stop()
      window.removeEventListener('keydown', down)
      window.removeEventListener('keyup', up)
      window.removeEventListener('blur', stop)
    }
  }, [active, onMove])

  return (
    <>
      <video ref={videoRef} autoPlay playsInline muted style={{ display: 'none' }} />
      <canvas ref={canvasRef} style={{ display: 'none' }} />
      {active && <GazeCrosshair gaze={gazePoint} />}
      {active && (
        <p role="status" aria-live="polite">
          {statusText} (arrow keys work too)
        </p>
      )}
      {active && !layoutMatches && (
        <button type="button" className="back-button" onClick={() => void enterFullscreen()}>
          Restore full screen
        </button>
      )}
    </>
  )
}
