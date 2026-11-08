import { afterEach, describe, expect, test, vi } from 'vitest'
import { gazeToDirection, getCameraIntrinsics, runGazeInference } from './gazeCapture'
import { predictGaze } from '../vision/model'

vi.mock('../vision/model', () => ({ predictGaze: vi.fn() }))
afterEach(() => { vi.restoreAllMocks(); vi.clearAllMocks() })

describe('gazeToDirection', () => {
  test('converts gaze coordinates into directions', () => {
    expect(gazeToDirection(0.1, 0.5)).toBe('left')
    expect(gazeToDirection(0.9, 0.5)).toBe('right')
    expect(gazeToDirection(0.5, 0.1)).toBe('up')
    expect(gazeToDirection(0.5, 0.9)).toBe('down')
  })

  test('stops movement when looking at the center', () => {
    expect(gazeToDirection(0.5, 0.5)).toBeNull()
  })

  test('rejects invalid gaze coordinates', () => {
    expect(gazeToDirection(NaN, 0.5)).toBeNull()
    expect(gazeToDirection(1.5, 0.5)).toBeNull()
  })
})

describe('getCameraIntrinsics', () => {
  test('calculates approximate camera intrinsics', () => {
    const image = {
      data: new Uint8Array(640 * 480 * 3),
      rows: 480,
      cols: 640,
    }

    expect(getCameraIntrinsics(image)).toEqual({
      focalLengthPx: 640,
      principalPoint: { x: 320, y: 240 },
    })
  })

  test('uses the same camera rule for portrait frames', () => {
    expect(getCameraIntrinsics({ data: new Uint8Array(), rows: 800, cols: 600 })).toEqual({
      focalLengthPx: 800, principalPoint: { x: 300, y: 400 },
    })
  })
})

describe('runGazeInference', () => {
  const landmarker = {} as Parameters<typeof runGazeInference>[2]
  const model = {} as Parameters<typeof runGazeInference>[3]

  test.each(['not ready', 'no dimensions', 'no context'])('skips an unavailable frame: %s', async (kind) => {
    const video = { readyState: kind === 'not ready' ? 1 : 2, videoWidth: kind === 'no dimensions' ? 0 : 2, videoHeight: 1 } as HTMLVideoElement
    const canvas = document.createElement('canvas')
    vi.spyOn(canvas, 'getContext').mockReturnValue(null)
    expect(await runGazeInference(video, canvas, landmarker, model)).toBeNull()
    expect(predictGaze).not.toHaveBeenCalled()
  })

  test.each([{ x: 0.2, y: 0.5 }, 'face_not_detected'] as const)('captures RGB with actual dimensions and returns the raw result: %j', async (result) => {
    const video = { readyState: 2, videoWidth: 2, videoHeight: 1 } as HTMLVideoElement
    const canvas = document.createElement('canvas')
    const drawImage = vi.fn()
    vi.spyOn(canvas, 'getContext').mockReturnValue({
      drawImage, getImageData: () => ({ data: new Uint8ClampedArray([255, 0, 128, 90, 3, 4, 5, 0]) }),
    } as unknown as CanvasRenderingContext2D)
    vi.mocked(predictGaze).mockResolvedValueOnce(result)
    expect(await runGazeInference(video, canvas, landmarker, model)).toEqual(result)
    expect([canvas.width, canvas.height]).toEqual([2, 1])
    expect(drawImage).toHaveBeenCalledWith(video, 0, 0, 2, 1)
    expect(predictGaze).toHaveBeenCalledExactlyOnceWith(landmarker, model,
      { data: new Uint8Array([255, 0, 128, 3, 4, 5]), rows: 1, cols: 2 },
      { focalLengthPx: 2, principalPoint: { x: 1, y: 0.5 } },
    )
  })
})
