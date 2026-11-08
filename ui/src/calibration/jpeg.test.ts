import { afterEach, beforeEach, expect, test, vi } from 'vitest'
import { decodeCalibrationJpeg } from './jpeg'

const close = vi.fn()
const drawImage = vi.fn()
const getImageData = vi.fn()
const bitmap = { width: 2, height: 1, close }
let canvases: HTMLCanvasElement[]

beforeEach(() => {
  canvases = []
  vi.stubGlobal('createImageBitmap', vi.fn().mockResolvedValue(bitmap))
  getImageData.mockReturnValue({
    data: new Uint8ClampedArray([255, 0, 10, 255, 20, 128, 30, 255]),
  })
  vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockImplementation(function (this: HTMLCanvasElement) {
    canvases.push(this)
    return { drawImage, getImageData } as unknown as CanvasRenderingContext2D
  })
})

afterEach(() => {
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
  vi.clearAllMocks()
})

test('decodes the original image dimensions and row-major RGB without mirroring', async () => {
  const blob = new Blob(['jpeg'], { type: 'image/jpeg' })
  const image = await decodeCalibrationJpeg(blob)

  expect(image).toEqual({
    rows: 1, cols: 2, data: new Uint8Array([255, 0, 10, 20, 128, 30]),
  })
  expect(createImageBitmap).toHaveBeenCalledWith(blob)
  expect(drawImage).toHaveBeenCalledWith(bitmap, 0, 0)
  expect(getImageData).toHaveBeenCalledWith(0, 0, 2, 1)
  expect(close).toHaveBeenCalledOnce()
  expect([canvases[0].width, canvases[0].height]).toEqual([0, 0])
})

test('releases the bitmap and canvas when pixel extraction fails', async () => {
  getImageData.mockImplementation(() => { throw new Error('pixel read failed') })
  await expect(decodeCalibrationJpeg(new Blob(['bad']))).rejects.toThrow('pixel read failed')
  expect(close).toHaveBeenCalledOnce()
  expect([canvases[0].width, canvases[0].height]).toEqual([0, 0])
})

test('releases the bitmap when no canvas context is available', async () => {
  vi.mocked(HTMLCanvasElement.prototype.getContext).mockReturnValue(null)
  await expect(decodeCalibrationJpeg(new Blob(['jpeg']))).rejects.toThrow('2D canvas')
  expect(close).toHaveBeenCalledOnce()
})

test('reports invalid image decoding without attempting pixel extraction', async () => {
  vi.mocked(createImageBitmap).mockRejectedValue(new Error('Invalid JPEG'))
  await expect(decodeCalibrationJpeg(new Blob(['bad']))).rejects.toThrow('Invalid JPEG')
  expect(drawImage).not.toHaveBeenCalled()
  expect(close).not.toHaveBeenCalled()
})
