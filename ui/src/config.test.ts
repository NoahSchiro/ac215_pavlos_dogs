import { afterEach, describe, expect, it, vi } from 'vitest'
import { DEFAULT_CONFIG, parseConfig } from './config'

describe('parseConfig', () => {
  it('reads a complete config', () => {
    expect(
      parseConfig({ modelUrl: 'https://storage.googleapis.com/b/models/r1/gazenet.onnx', telemetry: true, appVersion: 'abc123' }),
    ).toEqual({ modelUrl: 'https://storage.googleapis.com/b/models/r1/gazenet.onnx', telemetry: true, appVersion: 'abc123' })
  })

  it('accepts telemetry as the string nginx renders from an env var', () => {
    expect(parseConfig({ telemetry: 'true' }).telemetry).toBe(true)
    expect(parseConfig({ telemetry: 'false' }).telemetry).toBe(false)
  })

  it('falls back to the defaults for anything missing or malformed', () => {
    expect(parseConfig(null)).toEqual(DEFAULT_CONFIG)
    expect(parseConfig({ modelUrl: '  ', telemetry: 1, appVersion: 7 })).toEqual(DEFAULT_CONFIG)
  })
})

describe('loadRuntimeConfig', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
    vi.resetModules()
  })

  it('uses the defaults when the server answers with HTML (a dev server SPA fallback)', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response('<!doctype html>', { status: 200 })))
    const { loadRuntimeConfig } = await import('./config')
    expect(await loadRuntimeConfig()).toEqual(DEFAULT_CONFIG)
  })

  it('uses the served config when there is one, and fetches it once', async () => {
    const fetch = vi.fn(async () => Response.json({ modelUrl: '/m.onnx', telemetry: 'true' }))
    vi.stubGlobal('fetch', fetch)
    const { loadRuntimeConfig } = await import('./config')
    expect((await loadRuntimeConfig()).modelUrl).toBe('/m.onnx')
    await loadRuntimeConfig()
    expect(fetch).toHaveBeenCalledTimes(1)
  })

  it('never rejects, even when the network does', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => { throw new TypeError('offline') }))
    const { loadRuntimeConfig } = await import('./config')
    expect(await loadRuntimeConfig()).toEqual(DEFAULT_CONFIG)
  })
})
