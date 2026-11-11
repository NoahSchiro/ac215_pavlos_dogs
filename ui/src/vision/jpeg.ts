import type { ImageArray } from './utils'

/** Decode a captured JPEG at its original dimensions, without mirroring. */
export async function decodeJpeg(blob: Blob): Promise<ImageArray> {
  const bitmap = await createImageBitmap(blob)
  let canvas: HTMLCanvasElement | undefined
  try {
    canvas = document.createElement('canvas')
    canvas.width = bitmap.width
    canvas.height = bitmap.height
    const context = canvas.getContext('2d')
    if (!context) throw new Error('Could not get 2D canvas context for JPEG decoding')

    context.drawImage(bitmap, 0, 0)
    const rgba = context.getImageData(0, 0, bitmap.width, bitmap.height).data
    const rgb = new Uint8Array(bitmap.width * bitmap.height * 3)
    for (let i = 0, j = 0; i < rgb.length; i += 3, j += 4) {
      rgb[i] = rgba[j]
      rgb[i + 1] = rgba[j + 1]
      rgb[i + 2] = rgba[j + 2]
    }
    return { data: rgb, rows: bitmap.height, cols: bitmap.width }
  } finally {
    bitmap.close()
    if (canvas) {
      canvas.width = 0
      canvas.height = 0
    }
  }
}
