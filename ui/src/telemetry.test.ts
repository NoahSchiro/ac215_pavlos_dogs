import { afterEach, describe, expect, it, vi } from 'vitest'
import { GazeTelemetry, modelVersion } from './telemetry'

function setup() {
  const sent: Record<string, string>[] = []
  let clock = 1_000
  const telemetry = new GazeTelemetry(
    (params) => sent.push(Object.fromEntries(params)),
    { sessionId: 's1', model: 'run-7', appVersion: 'abc' },
    () => clock,
  )
  return { telemetry, sent, advance: (ms: number) => (clock += ms) }
}

describe('modelVersion', () => {
  it('names a published model by its directory', () => {
    expect(modelVersion('https://storage.googleapis.com/b/models/20261101-120000/gazenet.onnx')).toBe('20261101-120000')
    expect(modelVersion('https://storage.googleapis.com/b/models/v3/gazenet.onnx?x=1')).toBe('v3')
  })

  it("calls the copy shipped with the UI 'bundled'", () => {
    expect(modelVersion('/models/gazenet.onnx')).toBe('bundled')
    expect(modelVersion('gazenet.onnx')).toBe('bundled')
  })
})

describe('GazeTelemetry', () => {
  it('tags every beacon with the session, model and app version', () => {
    const { telemetry, sent } = setup()
    telemetry.start()
    expect(sent).toEqual([{ v: '1', ev: 'start', sid: 's1', model: 'run-7', app: 'abc' }])
    telemetry.recordLoadError()
    expect(sent[1]).toMatchObject({ ev: 'load_error', model: 'run-7' })
  })

  it('reports a window of frames as counts, rates and timings', () => {
    const { telemetry, sent, advance } = setup()
    telemetry.recordFrame('ok', 40)
    telemetry.recordFrame('ok', 60)
    telemetry.recordFrame('face_not_detected', 20)
    telemetry.recordFrame('error')
    advance(30_000)
    telemetry.flush()
    expect(sent[0]).toMatchObject({
      ev: 'frames', frames: '4', ok: '2', face_not_detected: '1', error: '1', pose_failed: '0',
      track_rate: '0.5', lat_mean: '40', lat_max: '60', window_s: '30',
    })
  })

  it('starts a fresh window after each flush, and skips empty ones', () => {
    const { telemetry, sent } = setup()
    telemetry.recordFrame('ok', 10)
    telemetry.flush()
    telemetry.flush()
    telemetry.recordFrame('pose_failed', 30)
    telemetry.flush()
    expect(sent.map((s) => [s.frames, s.ok, s.pose_failed])).toEqual([['1', '1', '0'], ['1', '0', '1']])
  })

  it('never puts a gaze coordinate or image data on the wire', () => {
    const { telemetry, sent } = setup()
    telemetry.start()
    telemetry.recordFrame('ok', 12)
    telemetry.flush()
    const keys = new Set(sent.flatMap(Object.keys))
    for (const forbidden of ['x', 'y', 'gaze', 'image', 'landmarks']) expect(keys.has(forbidden)).toBe(false)
  })
})

describe('startTelemetry', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
    vi.resetModules()
    vi.useRealTimers()
  })

  async function start(config: object) {
    vi.useFakeTimers()
    vi.stubGlobal('fetch', vi.fn(async () => Response.json(config)))
    const sent: Record<string, string>[] = []
    const { startTelemetry } = await import('./telemetry')
    const { reportInference } = await import('./vision/events')
    await startTelemetry((params) => sent.push(Object.fromEntries(params)))
    return { sent, reportInference }
  }

  it('turns the pipeline events into beacons for the configured model', async () => {
    const { sent, reportInference } = await start({
      telemetry: 'true',
      modelUrl: 'https://storage.googleapis.com/b/models/run-42/gazenet.onnx',
      appVersion: 'abc',
    })
    reportInference({ kind: 'load_start' })
    reportInference({ kind: 'frame', outcome: 'ok', latencyMs: 80 })
    reportInference({ kind: 'frame', outcome: 'face_not_detected', latencyMs: 20 })
    reportInference({ kind: 'load_error' })
    vi.advanceTimersByTime(30_000)
    expect(sent.map((s) => s.ev)).toEqual(['start', 'load_error', 'frames'])
    expect(sent[2]).toMatchObject({ model: 'run-42', app: 'abc', frames: '2', ok: '1', track_rate: '0.5' })
  })

  it('stays silent unless the deployment enables it', async () => {
    const { sent, reportInference } = await start({ telemetry: 'false' })
    reportInference({ kind: 'load_start' })
    reportInference({ kind: 'frame', outcome: 'ok', latencyMs: 80 })
    vi.advanceTimersByTime(60_000)
    expect(sent).toEqual([])
  })
})
