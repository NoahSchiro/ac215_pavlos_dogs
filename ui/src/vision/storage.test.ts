import { afterEach, expect, test, vi } from 'vitest'
import { CALIBRATION_POINTS } from '../calibration/points'
import { createCalibrationRecord } from '../calibration/record'
import { fitCalibration } from '../calibration/ridge'
import { clearCalibration, loadCalibration, saveCalibration } from '../calibration/storage'

// This origin is the Vitest server, separate from the user's dev-app origin.
let frame: HTMLIFrameElement | undefined
afterEach(async () => { frame?.remove(); frame = undefined; await clearCalibration() })

test('a new app document restores real IndexedDB JPEGs/fit and can clear them', async () => {
  await clearCalibration()
  const canvas = document.createElement('canvas')
  canvas.width = 640
  canvas.height = 480
  canvas.getContext('2d')!.fillRect(0, 0, 640, 480)
  const jpeg = await new Promise<Blob>((resolve, reject) => {
    canvas.toBlob((blob) => blob ? resolve(blob) : reject(new Error('JPEG encoding failed')), 'image/jpeg')
  })
  canvas.width = canvas.height = 0
  const samples = CALIBRATION_POINTS.map((target, pointIndex) => ({
    image: jpeg, target, pointIndex, timestamp: 1, viewportWidth: 800, viewportHeight: 600,
  }))
  const record = createCalibrationRecord({
    capture: {
      samples, completedAt: 2,
      camera: { deviceId: 'browser-test-camera', label: 'Fixture', width: 640, height: 480, facingMode: 'user' },
    },
    frames: samples.map((sample) => ({ sample, result: {
      ok: true as const, prediction: sample.target, image: { width: 640, height: 480 },
      intrinsics: { focalLengthPx: 640, principalPoint: { x: 320, y: 240 } },
    } })),
    validFrameCounts: Array(9).fill(1),
    fit: fitCalibration(CALIBRATION_POINTS, CALIBRATION_POINTS),
  })
  await saveCalibration(record)
  const loaded = await loadCalibration()
  if (loaded.status !== 'restored') throw new Error(`Expected restored record: ${loaded.status}`)
  expect(loaded.record.calibration.fit).toEqual(record.calibration.fit)
  expect(Array.from(new Uint8Array(await loaded.record.calibration.capture.samples[0].image.arrayBuffer())))
    .toEqual(Array.from(new Uint8Array(await jpeg.arrayBuffer())))
  frame = document.createElement('iframe')
  frame.src = '/index.html'
  document.body.appendChild(frame)
  await vi.waitFor(() => {
    expect(frame!.contentDocument?.body.textContent).toContain('Saved calibration restored.')
  }, { timeout: 10000 })
  await new Promise<void>((resolve) => {
    frame!.addEventListener('load', () => resolve(), { once: true })
    frame!.contentWindow!.location.reload()
  })
  await vi.waitFor(() => {
    expect(frame!.contentDocument?.body.textContent).toContain('Saved calibration restored.')
  }, { timeout: 10000 })
  expect(frame.contentDocument!.body.textContent).toContain('9 valid frames')
  const button = Array.from(frame.contentDocument!.querySelectorAll('button')).find((item) => item.textContent === 'Clear calibration')!
  button.click()
  await vi.waitFor(() => { expect(frame!.contentDocument?.body.textContent).toContain('Calibration cleared.') })
  expect(await loadCalibration()).toEqual({ status: 'empty' })
}, 15000)
