import type { CameraIntrinsics, ImageArray } from '../vision/utils'

/** Drop the alpha channel: canvas pixels are RGBA, the model takes RGB. */
function rgbaToRgb(rgba: Uint8ClampedArray, width: number, height: number): ImageArray {
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

  return rgbaToRgb(context.getImageData(0, 0, width, height).data, width, height)
}

/** Decode a captured frame (calibration stores them as JPEG) for gaze inference. */
export async function decodeGazeFrame(blob: Blob): Promise<ImageArray> {
  const bitmap = await createImageBitmap(blob)
  const { width, height } = bitmap
  const canvas = new OffscreenCanvas(width, height)
  const context = canvas.getContext('2d')
  if (!context) throw new Error('Could not get 2D canvas context')
  context.drawImage(bitmap, 0, 0)
  bitmap.close()
  return rgbaToRgb(context.getImageData(0, 0, width, height).data, width, height)
}

/**
 * Approximate webcam intrinsics; the actual focal length varies by device.
 *
 * Calibration frames and live frames must use the same guess: the per-user
 * fit then absorbs whatever error the guess introduces, along with the rest
 * of the user's bias.
 */
export function estimateIntrinsics(image: ImageArray): CameraIntrinsics {
  return {
    focalLengthPx: Math.max(image.cols, image.rows),
    principalPoint: {
      x: image.cols / 2,
      y: image.rows / 2,
    },
  }
}
