import type { ImageArray } from '../vision/utils'
import { predictGaze } from '../vision/model'

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

/** Approximate webcam camera intrinsics from image dimensions. */
export function getCameraIntrinsics(image: ImageArray) {
  return {
    focalLengthPx: Math.max(image.cols, image.rows),
    principalPoint: {
      x: image.cols / 2,
      y: image.rows / 2,
    },
  }
}

/** Convert normalized gaze coordinates into a maze movement direction. */
export type Direction = 'up' | 'down' | 'left' | 'right'

export function gazeToDirection(x: number, y: number): Direction | null {
  if (!Number.isFinite(x) || !Number.isFinite(y)) return null
  if (x < 0 || x > 1 || y < 0 || y > 1) return null

  const CENTER_MIN = 0.4
  const CENTER_MAX = 0.6

  const dx = x < CENTER_MIN ? x - CENTER_MIN : x > CENTER_MAX ? x - CENTER_MAX : 0
  const dy = y < CENTER_MIN ? y - CENTER_MIN : y > CENTER_MAX ? y - CENTER_MAX : 0

  if (dx === 0 && dy === 0) return null
  if (Math.abs(dx) > Math.abs(dy)) return dx < 0 ? 'left' : 'right'
  return dy < 0 ? 'up' : 'down'
}

/** Run one gaze prediction using a captured webcam frame. */
export async function runGazeInference(
  video: HTMLVideoElement,
  canvas: HTMLCanvasElement,
  landmarker: Parameters<typeof predictGaze>[0],
  model: Parameters<typeof predictGaze>[1],
): Promise<Awaited<ReturnType<typeof predictGaze>> | null> {
  if (video.readyState < 2) return null

  const image = captureGazeFrame(video, canvas)
  if (!image) return null

  return predictGaze(
    landmarker,
    model,
    image,
    getCameraIntrinsics(image),
  )
}
