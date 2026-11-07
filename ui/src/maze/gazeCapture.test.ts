import { describe, expect, test } from 'vitest'
import { gazeToDirection, getCameraIntrinsics } from './gazeCapture'

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
})
