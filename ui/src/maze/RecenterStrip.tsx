import type { Ref } from 'react'
import type { RecenterState } from './recenter'

interface Props {
  state: RecenterState
  /** How far a look at the strip is toward starting recentering, 0 to 1. */
  progress: number
  onClick: () => void
  columnRef: Ref<HTMLDivElement>
  stripRef: Ref<HTMLButtonElement>
}

function label(state: RecenterState): string {
  switch (state.phase) {
    case 'confirming':
    case 'capturing':
      return 'Now stare at Pavlos'
    case 'done':
      return 'Recentered'
    default:
      return 'Look here to recenter'
  }
}

/**
 * The Recenter strip in the column beside the maze. Looking at it fills it;
 * full starts recentering. A click (or R) works too.
 */
export function RecenterStrip({ state, progress, onClick, columnRef, stripRef }: Props) {
  const active = state.phase === 'confirming' || state.phase === 'capturing'
  return (
    <div className="maze-recenter-column" ref={columnRef}>
      <button
        type="button"
        ref={stripRef}
        className="maze-recenter"
        data-active={active}
        data-testid="recenter-strip"
        onClick={onClick}
      >
        <span className="maze-recenter-fill" style={{ height: `${progress * 100}%` }} aria-hidden="true" />
        <span className="maze-recenter-icon" aria-hidden="true">◎</span>
        <span className="maze-recenter-label">{label(state)}</span>
      </button>
    </div>
  )
}
