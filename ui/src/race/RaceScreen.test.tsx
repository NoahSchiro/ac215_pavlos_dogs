import { act, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { RaceScreen } from './RaceScreen'
import type { RaceMazeData } from './raceClient'

// The camera/model loop is tested elsewhere; here gaze is simply "not looking anywhere".
vi.mock('../gaze/useGazeTracking', () => ({
  useGazeTracking: () => ({
    videoRef: { current: null },
    canvasRef: { current: null },
    gazeDirection: null,
    gazePoint: null,
    statusText: 'Tracking gaze',
  }),
}))

/** A stand-in for the browser WebSocket that lets the test play the server. */
class FakeSocket {
  static OPEN = 1
  static instances: FakeSocket[] = []
  readyState = FakeSocket.OPEN
  sent: string[] = []
  closed = false
  onmessage: ((event: { data: string }) => void) | null = null
  onerror: (() => void) | null = null
  url: string
  constructor(url: string) {
    this.url = url
    FakeSocket.instances.push(this)
  }
  send(data: string) {
    this.sent.push(data)
  }
  close() {
    this.closed = true
  }
  receive(message: unknown) {
    act(() => this.onmessage?.({ data: JSON.stringify(message) }))
  }
}

const open = { top: false, right: false, bottom: false, left: false }
const maze: RaceMazeData = {
  rows: 2,
  cols: 2,
  walls: [
    [open, open],
    [open, open],
  ],
  start: { row: 0, col: 0 },
  cheese: { row: 1, col: 1 },
}

function state(players: Record<string, { row: number; col: number }>, winner: number | null = null) {
  return { type: 'state', started: Object.keys(players).length === 2, players, winner }
}

describe('RaceScreen', () => {
  beforeEach(() => {
    FakeSocket.instances = []
    vi.stubGlobal('WebSocket', FakeSocket)
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({ ok: true, json: async () => ({ code: 'ABCD' }) }),
    )
  })
  afterEach(() => vi.unstubAllGlobals())

  async function createAndJoin() {
    render(<RaceScreen onBack={() => {}} serverUrl="http://server.test" />)
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: 'Create race' }))
    })
    const socket = FakeSocket.instances[0]
    expect(socket.url).toBe('ws://server.test/rooms/ABCD/ws')
    socket.receive({ type: 'welcome', player: 1, maze })
    return socket
  }

  it('creates a race and shows the code to share', async () => {
    const socket = await createAndJoin()
    socket.receive(state({ '1': { row: 0, col: 0 } }))
    expect(screen.getByLabelText('Your race code')).toHaveTextContent('ABCD')
  })

  it('shows the opponent as a ghost and sends arrow-key moves', async () => {
    const socket = await createAndJoin()
    socket.receive(state({ '1': { row: 0, col: 0 }, '2': { row: 0, col: 1 } }))
    expect(screen.getByTestId('race-me')).toBeInTheDocument()
    expect(screen.getByTestId('race-ghost')).toBeInTheDocument()

    fireEvent.keyDown(window, { key: 'ArrowDown' })
    fireEvent.keyUp(window, { key: 'ArrowDown' })
    expect(socket.sent.map((m) => JSON.parse(m))).toEqual([{ type: 'move', direction: 'down' }])
  })

  it('announces the winner', async () => {
    const socket = await createAndJoin()
    socket.receive(state({ '1': { row: 1, col: 1 }, '2': { row: 0, col: 1 } }, 1))
    expect(screen.getByRole('status')).toHaveTextContent('You found the cheese first!')
  })

  it('tells you when the opponent leaves', async () => {
    const socket = await createAndJoin()
    socket.receive(state({ '1': { row: 0, col: 0 }, '2': { row: 0, col: 1 } }))
    socket.receive({ type: 'opponent_left' })
    expect(screen.getByText('Your opponent left the race.')).toBeInTheDocument()
  })

  it('shows an error for a bad join code', () => {
    render(<RaceScreen onBack={() => {}} serverUrl="http://server.test" />)
    fireEvent.change(screen.getByLabelText('Race code'), { target: { value: 'zzzz' } })
    fireEvent.click(screen.getByRole('button', { name: 'Join' }))
    const socket = FakeSocket.instances[0]
    expect(socket.url).toBe('ws://server.test/rooms/ZZZZ/ws')
    socket.receive({ type: 'error', reason: 'room_not_found' })
    expect(screen.getByRole('alert')).toHaveTextContent('No race with that code')
  })

  it('closes the connection when leaving the screen', async () => {
    const { unmount } = render(<RaceScreen onBack={() => {}} serverUrl="http://server.test" />)
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: 'Create race' }))
    })
    unmount()
    expect(FakeSocket.instances[0].closed).toBe(true)
  })
})

describe('RaceMaze rolling', () => {
  it('rolls Pavlos a quarter turn per move, like the solo maze', async () => {
    const { RaceMaze } = await import('./RaceMaze')
    const { rerender } = render(
      <RaceMaze maze={maze} me={{ row: 0, col: 0 }} ghost={null} showCheese />,
    )
    const img = () => screen.getByAltText('You')
    expect(img()).toHaveAttribute('src', '/pavlos_smile.png')

    rerender(<RaceMaze maze={maze} me={{ row: 0, col: 1 }} ghost={null} showCheese />)
    expect(img().style.transform).toBe('rotate(90deg)')
    expect(img()).toHaveAttribute('src', '/pavlos_frown.png')

    rerender(<RaceMaze maze={maze} me={{ row: 0, col: 0 }} ghost={null} showCheese />)
    expect(img().style.transform).toBe('rotate(0deg)')
  })
})
