import type { ImageArray } from '../vision/utils'
import { predictGaze } from '../vision/model'
import { approximateCameraIntrinsics } from '../vision/camera'

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
    approximateCameraIntrinsics(image.cols, image.rows),
  )
}
