import type { CSSProperties } from 'react'
import type { MazePosition } from './mazeGenerator'
import type { RecenterState } from './recenter'

interface Props {
  state: RecenterState
  /** Stare progress, 0 to 1. */
  progress: number
  /** The user is holding a stare: show the ring only, no text to read. */
  held: boolean
  player: MazePosition
  rows: number
  cols: number
}

function message(state: RecenterState, held: boolean): string | null {
  switch (state.phase) {
    case 'confirming':
      if (held) return null
      return state.retry ? 'Missed, stare at me again' : 'Stare at me'
    case 'done':
      return 'Done ✓'
    case 'gaveUp':
      return "Let's try later"
    default:
      return null
  }
}

/**
 * What Pavlos says during recentering, and a progress ring around him. The
 * bubble disappears once a stare is held, because reading it would pull the
 * stare off Pavlos and skew the measured offset.
 */
export function RecenterCue({ state, progress, held, player, rows, cols }: Props) {
  if (state.phase === 'idle') return null
  const text = message(state, held)
  const showRing = state.phase === 'confirming' || state.phase === 'capturing'
  const left = `${((player.col + 0.5) / cols) * 100}%`
  // Speak from above, or from below on the top row; keep clear of the side edges.
  const below = player.row === 0
  const shiftX = player.col === 0 ? '-20%' : player.col === cols - 1 ? '-80%' : '-50%'

  return (
    <>
      {showRing && (
        <div
          className={state.phase === 'capturing' ? 'recenter-ring capturing' : 'recenter-ring'}
          data-testid="recenter-ring"
          aria-hidden="true"
          style={{
            left,
            top: `${((player.row + 0.5) / rows) * 100}%`,
            width: `${130 / cols}%`,
            height: `${130 / rows}%`,
          }}
        >
          <svg viewBox="0 0 100 100">
            <circle className="recenter-ring-track" cx="50" cy="50" r="45" pathLength={1} />
            <circle
              className="recenter-ring-fill"
              cx="50"
              cy="50"
              r="45"
              pathLength={1}
              // A round cap would draw a dot at zero progress.
              style={{ '--progress': progress, opacity: progress > 0 ? 1 : 0 } as CSSProperties}
            />
          </svg>
        </div>
      )}
      <div className="recenter-bubble-anchor" role="status" aria-live="polite">
        {text && (
          <div
            className="recenter-bubble"
            data-testid="recenter-bubble"
            style={{
              left,
              top: `${((below ? player.row + 1 : player.row) / rows) * 100}%`,
              transform: `translate(${shiftX}, ${below ? '8px' : 'calc(-100% - 8px)'})`,
            }}
          >
            {text}
          </div>
        )}
      </div>
    </>
  )
}
