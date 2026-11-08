import { expect, test } from 'vitest'
import { approximateCameraIntrinsics } from '../vision/camera'

test.each([
  [640, 480, 640, 320, 240],
  [480, 640, 640, 240, 320],
  [1280, 720, 1280, 640, 360],
])('approximates camera intrinsics from a %ix%i image', (width, height, focalLengthPx, x, y) => {
  expect(approximateCameraIntrinsics(width, height)).toEqual({
    focalLengthPx, principalPoint: { x, y },
  })
})

test.each([0, -1, NaN, Infinity, 1.5])('rejects an invalid image dimension: %s', (dimension) => {
  expect(() => approximateCameraIntrinsics(dimension, 480)).toThrow('positive integers')
  expect(() => approximateCameraIntrinsics(640, dimension)).toThrow('positive integers')
})
