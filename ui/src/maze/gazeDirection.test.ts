import { afterEach, expect, test, vi } from 'vitest'
import { TEST_CAMERA, testSession } from '../calibration/fixtures/testSession'
import * as ridge from '../calibration/ridge'
import type { CalibrationSession } from '../calibration/record'
import { gazeToDirection, prepareGameplayGaze } from './gazeDirection'

const dimensions = { cols: 640, rows: 480 }
afterEach(() => vi.restoreAllMocks())

test('applies the real affine correction once, after the raw prediction', () => {
  const session = testSession()
  const apply = vi.spyOn(ridge, 'applyCalibration')
  const result = prepareGameplayGaze({ x: 0.2, y: 0.5 }, session, TEST_CAMERA, dimensions)!
  expect(result.calibrated).toBe(true)
  expect(result.point.x).toBeCloseTo(0.8, 12)
  expect(gazeToDirection(result.point.x, result.point.y)).toBe('right')
  expect(apply).toHaveBeenCalledTimes(1)
})

test.each(['missing', 'pending', 'camera', 'size', 'unknown'] as const)('uses the unchanged raw prediction for %s calibration', (kind) => {
  const session = testSession()
  const apply = vi.spyOn(ridge, 'applyCalibration')
  const raw = { x: 0.2, y: 0.5 }
  const result = prepareGameplayGaze(raw,
    kind === 'missing' ? null : { ...session, cameraVerification: kind === 'pending' ? 'pending' : 'verified' },
    kind === 'unknown' ? null : { ...TEST_CAMERA, deviceId: kind === 'camera' ? 'other' : TEST_CAMERA.deviceId },
    kind === 'size' ? { cols: 1280, rows: 720 } : dimensions,
  )!
  expect(result).toEqual({ point: raw, calibrated: false })
  expect(gazeToDirection(result.point.x, result.point.y)).toBe('left')
  expect(apply).not.toHaveBeenCalled()
})

test('clamps finite corrected coordinates at the gameplay boundary', () => {
  const session = testSession()
  const adjusted = { ...session, record: { ...session.record, calibration: {
    ...session.record.calibration, fit: { weight: [[1, 0], [0, 1]] as const, intercept: [2, -2] as const, lambda: 0.1 },
  } } }
  expect(prepareGameplayGaze({ x: 0.5, y: 0.5 }, adjusted, TEST_CAMERA, dimensions)).toEqual({
    calibrated: true, point: { x: 1, y: 0 },
  })
})

test.each([{ x: NaN, y: 0.5 }, { x: 0.5, y: Infinity }])('rejects non-finite raw output: %j', (point) => {
  expect(prepareGameplayGaze(point, testSession(), TEST_CAMERA, dimensions)).toBeNull()
})

test('rejects a non-finite correction rather than clamping it to a movement', () => {
  const session = testSession()
  const invalid = { ...session, record: { ...session.record, calibration: {
    ...session.record.calibration, fit: { weight: [[Infinity, 0], [0, 1]] as const, intercept: [0, 0] as const, lambda: 0.1 },
  } } }
  expect(prepareGameplayGaze({ x: 0.5, y: 0.5 }, invalid, TEST_CAMERA, dimensions)).toBeNull()
})

test.each(['model', 'format'] as const)('does not use a verified fit after its %s version changes', (kind) => {
  const session = testSession()
  const incompatible = { ...session, record: {
    ...session.record, version: kind === 'format' ? 99 : 1,
    compatibility: { ...session.record.compatibility, modelId: kind === 'model' ? 'old-model' : session.record.compatibility.modelId },
  } } as unknown as CalibrationSession
  expect(prepareGameplayGaze({ x: 0.2, y: 0.5 }, incompatible, TEST_CAMERA, dimensions)).toEqual({
    calibrated: false, point: { x: 0.2, y: 0.5 },
  })
})
