import { afterEach, expect, test, vi } from 'vitest'
import { CAMERA_CONSTRAINTS, captureCameraSettings, verifyCaptureCamera } from './cameraMetadata'

const camera = { deviceId: 'camera-a', label: 'Camera A', width: 640, height: 480, facingMode: 'user' }
function stream(settings: MediaTrackSettings = camera) {
  const track = { label: 'Camera A', getSettings: () => settings, stop: vi.fn() }
  return { track, media: { getVideoTracks: () => [track], getTracks: () => [track] } as unknown as MediaStream }
}
afterEach(() => vi.restoreAllMocks())

test('captures actual track identity and dimensions', () => {
  expect(captureCameraSettings(stream().media)).toEqual(camera)
})
test('does not invent a missing camera identity', () => {
  expect(captureCameraSettings(stream({ width: 640, height: 480 }).media)).toBeNull()
  expect(captureCameraSettings(null)).toBeNull()
})
test('verifies the selected camera and always stops its tracks', async () => {
  const current = stream()
  const getUserMedia = vi.fn().mockResolvedValue(current.media)
  Object.defineProperty(navigator, 'mediaDevices', { configurable: true, value: { getUserMedia } })
  await verifyCaptureCamera(camera)
  expect(getUserMedia).toHaveBeenCalledWith(CAMERA_CONSTRAINTS)
  expect(current.track.stop).toHaveBeenCalledOnce()
})
test.each([
  { ...camera, deviceId: 'camera-b' },
  { ...camera, width: 1280 },
  { ...camera, height: 720 },
  { ...camera, deviceId: '' },
])('rejects unknown/mismatched cameras and releases the stream: %j', async (settings) => {
  const current = stream(settings)
  Object.defineProperty(navigator, 'mediaDevices', { configurable: true, value: { getUserMedia: vi.fn().mockResolvedValue(current.media) } })
  await expect(verifyCaptureCamera(camera)).rejects.toThrow(/recalibrat/i)
  expect(current.track.stop).toHaveBeenCalledOnce()
})
test('releases a camera that resolves after cancellation', async () => {
  const current = stream()
  const controller = new AbortController()
  let finish!: (media: MediaStream) => void
  Object.defineProperty(navigator, 'mediaDevices', { configurable: true, value: {
    getUserMedia: vi.fn().mockReturnValue(new Promise((resolve) => { finish = resolve })),
  } })
  const pending = verifyCaptureCamera(camera, controller.signal)
  const rejected = expect(pending).rejects.toMatchObject({ name: 'AbortError' })
  controller.abort()
  finish(current.media)
  await rejected
  expect(current.track.stop).toHaveBeenCalledOnce()
})
