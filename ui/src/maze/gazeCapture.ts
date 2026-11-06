import type { ImageArray } from '../vision/utils'

/** Convert a webcam video frame into an RGB image for gaze inference. */
export function captureGazeFrame(
  video: HTMLVideoElement,
  canvas: HTMLCanvasElement,
): ImageArray | null {
  const width = video.videoWidth
  const height = video.videoHeight

  if (!width || !height) return null

  canvas.width = width
  canvas.height = height

  const context = canvas.getContext('2d', { willReadFrequently: true })
  if (!context) return null

  context.drawImage(video, 0, 0, width, height)

  const rgba = context.getImageData(0, 0, width, height).data
  const rgb = new Uint8Array(width * height * 3)

  for (let i = 0, j = 0; i < rgba.length; i += 4, j += 3) {
    rgb[j] = rgba[i]
    rgb[j + 1] = rgba[i + 1]
    rgb[j + 2] = rgba[i + 2]
  }

  return {
    data: rgb,
    rows: height,
    cols: width,
  }
}
