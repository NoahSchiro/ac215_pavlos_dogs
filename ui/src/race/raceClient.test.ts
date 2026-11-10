import { afterEach, describe, expect, it, vi } from 'vitest'
import { createRoom, moveMessage, roomSocketUrl } from './raceClient'

describe('raceClient', () => {
  afterEach(() => vi.unstubAllGlobals())

  it('turns the server URL into a WebSocket URL for the room', () => {
    expect(roomSocketUrl('http://localhost:8000', 'abcd')).toBe('ws://localhost:8000/rooms/ABCD/ws')
    expect(roomSocketUrl('https://race.example.com/', 'WXYZ')).toBe(
      'wss://race.example.com/rooms/WXYZ/ws',
    )
  })

  it('creates a room and returns its code', async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => ({ code: 'QRST' }) })
    vi.stubGlobal('fetch', fetchMock)
    await expect(createRoom('http://localhost:8000')).resolves.toBe('QRST')
    expect(fetchMock).toHaveBeenCalledWith('http://localhost:8000/rooms', { method: 'POST' })
  })

  it('reports a failed room creation', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: false, status: 500 }))
    await expect(createRoom('http://localhost:8000')).rejects.toThrow()
  })

  it('encodes a move', () => {
    expect(JSON.parse(moveMessage('left'))).toEqual({ type: 'move', direction: 'left' })
  })
})
