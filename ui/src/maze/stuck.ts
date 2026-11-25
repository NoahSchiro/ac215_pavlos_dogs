import type { MazePosition } from './mazeGenerator'

export interface StuckSettings {
  /** Stuck if Pavlos stays in one cell this long while the user is trying to steer. */
  stillMs: number
  /** Window for spotting back-and-forth rocking. */
  windowMs: number
  /** Rocking: at least this many moves in the window... */
  rockMoves: number
  /** ...that visit no more than this many distinct cells. */
  rockCells: number
  /** How long the prompt shows. */
  promptMs: number
  /** Quiet time after a prompt ends (or recentering), before prompting again. */
  cooldownMs: number
}

export interface StuckState {
  /** Pavlos's cell at the last sample. */
  cell: MazePosition | null
  /** When Pavlos arrived in `cell`. */
  stillSince: number | null
  /** When the current run of "trying to steer" samples began. */
  tryingSince: number | null
  /** Recent moves, within `windowMs`. */
  moves: Array<{ at: number; cell: MazePosition }>
  /** When the current prompt began, or null if not prompting. */
  promptSince: number | null
  /** No new prompt before this time. */
  quietUntil: number
}

export const NEW_STUCK: StuckState = {
  cell: null, stillSince: null, tryingSince: null, moves: [], promptSince: null, quietUntil: 0,
}

export interface StuckSample {
  now: number
  player: MazePosition
  /**
   * The user seems to be steering: gaze is near Pavlos (inside the pull ring)
   * but not on his own cell, which would mean stopping on purpose.
   */
  trying: boolean
}

const sameCell = (a: MazePosition, b: MazePosition) => a.row === b.row && a.col === b.col

/**
 * Add one gaze sample and decide whether to prompt "Stuck? Recenter". Two
 * signs of being stuck: Pavlos stays in one cell for `stillMs` while the user
 * keeps trying to steer, or he rocks back and forth between two cells. A
 * prompt lasts `promptMs`, then none comes for `cooldownMs`.
 */
export function trackStuck(state: StuckState, sample: StuckSample, settings: StuckSettings): StuckState {
  const { now, player, trying } = sample
  let { cell, stillSince, tryingSince, moves, promptSince, quietUntil } = state

  if (!cell || !sameCell(cell, player)) {
    if (cell) moves = [...moves, { at: now, cell: player }]
    cell = player
    stillSince = now
  }
  moves = moves.filter((move) => now - move.at <= settings.windowMs)
  tryingSince = trying ? (tryingSince ?? now) : null

  const next = { cell, stillSince, tryingSince, moves, promptSince, quietUntil }
  if (promptSince !== null) {
    if (now - promptSince < settings.promptMs) return next
    return { ...next, promptSince: null, quietUntil: now + settings.cooldownMs }
  }
  if (now < quietUntil) return next

  const stillStuck = tryingSince !== null && stillSince !== null &&
    now - Math.max(tryingSince, stillSince) >= settings.stillMs
  const visited = new Set(moves.map((move) => `${move.cell.row},${move.cell.col}`))
  const rocking = moves.length >= settings.rockMoves && visited.size <= settings.rockCells
  if (!stillStuck && !rocking) return next

  // Prompt, and start the evidence afresh.
  return { ...next, moves: [], tryingSince: null, stillSince: now, promptSince: now }
}

/** True while the "Stuck?" prompt should show. */
export function isStuckPrompt(state: StuckState): boolean {
  return state.promptSince !== null
}

/** End any prompt and stay quiet for `cooldownMs`, e.g. while and after recentering. */
export function quietStuck(state: StuckState, now: number, settings: StuckSettings): StuckState {
  return { ...state, moves: [], tryingSince: null, promptSince: null, quietUntil: now + settings.cooldownMs }
}
