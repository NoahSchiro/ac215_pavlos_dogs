/**
 * Talking to the race backend: create a room over HTTP, then play over a WebSocket.
 * The message shapes here mirror race_server/app/main.py.
 */

import type { Direction } from '../maze/gazeCapture'

export interface Cell {
  row: number
  col: number
}

export interface Walls {
  top: boolean
  right: boolean
  bottom: boolean
  left: boolean
}

export interface RaceMazeData {
  rows: number
  cols: number
  walls: Walls[][]
  start: Cell
  cheese: Cell
}

/** Messages the server sends. */
export type ServerMessage =
  | { type: 'welcome'; player: number; maze: RaceMazeData }
  | { type: 'state'; started: boolean; players: Record<string, Cell>; winner: number | null }
  | { type: 'rejected'; reason: string }
  | { type: 'opponent_left' }
  | { type: 'error'; reason: string }

/** Ask the server for a new room; resolves to its code. */
export async function createRoom(baseUrl: string): Promise<string> {
  const response = await fetch(`${baseUrl}/rooms`, { method: 'POST' })
  if (!response.ok) throw new Error(`Could not create a race (HTTP ${response.status})`)
  const body = (await response.json()) as { code: string }
  return body.code
}

/** The WebSocket address for a room: http(s):// becomes ws(s)://. */
export function roomSocketUrl(baseUrl: string, code: string): string {
  const url = new URL(`/rooms/${encodeURIComponent(code.trim().toUpperCase())}/ws`, baseUrl)
  url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:'
  return url.toString()
}

export function moveMessage(direction: Direction): string {
  return JSON.stringify({ type: 'move', direction })
}
