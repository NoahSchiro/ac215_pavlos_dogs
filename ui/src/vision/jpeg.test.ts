import { expect, test } from 'vitest'
import { decodeCalibrationJpeg } from '../calibration/jpeg'
import { predictCalibrationImage } from '../calibration/inference'

test('decodes a real browser JPEG with the original colors and orientation', async () => {
  const canvas = document.createElement('canvas')
  canvas.width = 32
  canvas.height = 16
  const context = canvas.getContext('2d')!
  context.fillStyle = 'red'
  context.fillRect(0, 0, 16, 16)
  context.fillStyle = 'blue'
  context.fillRect(16, 0, 16, 16)
  const blob = await new Promise<Blob>((resolve, reject) => {
    canvas.toBlob((result) => result ? resolve(result) : reject(new Error('JPEG encoding failed')), 'image/jpeg', 1)
  })
  canvas.width = canvas.height = 0

  const image = await decodeCalibrationJpeg(blob)
  expect([image.rows, image.cols, image.data.length]).toEqual([16, 32, 16 * 32 * 3])
  const left = (8 * 32 + 4) * 3
  const right = (8 * 32 + 28) * 3
  // JPEG is lossy; check channel dominance away from the boundary.
  expect(image.data[left]).toBeGreaterThan(240)
  expect(image.data[left + 2]).toBeLessThan(15)
  expect(image.data[right]).toBeLessThan(15)
  expect(image.data[right + 2]).toBeGreaterThan(240)
})

test('rejects corrupt JPEG bytes in the real browser decoder', async () => {
  await expect(decodeCalibrationJpeg(new Blob(['invalid'], { type: 'image/jpeg' }))).rejects.toThrow()
})

test('the callable diagnostic exposes a decoding failure in the browser', async () => {
  const result = await predictCalibrationImage(new Blob(['invalid'], { type: 'image/jpeg' }))
  expect(result).toMatchObject({ ok: false, stage: 'decode', reason: expect.any(String) })
  if (!result.ok) expect(result.reason.length).toBeGreaterThan(0)
})
