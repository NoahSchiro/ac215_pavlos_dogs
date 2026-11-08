import { act, fireEvent, render, screen } from '@testing-library/react'
import { beforeEach, expect, test, vi } from 'vitest'
import App from '../App'
import { captureBurst } from './capture'
import { predictCalibrationImage, type CalibrationImagePrediction } from './inference'
import { CALIBRATION_POINTS } from './points'
import { useCamera } from './useCamera'
import { loadCalibration, saveCalibration } from './storage'

// Exercise the actual capture screen, App, processing, and ridge fit together.
// Only camera/encoding, fullscreen, and model inference need browser resources.
vi.mock('./capture', () => ({ captureBurst: vi.fn() }))
vi.mock('./inference', () => ({ predictCalibrationImage: vi.fn() }))
vi.mock('./useCamera', () => ({ useCamera: vi.fn() }))
vi.mock('./storage', () => ({ loadCalibration: vi.fn(), saveCalibration: vi.fn(), clearCalibration: vi.fn() }))
vi.mock('./fullscreen', () => ({
  enterFullscreen: vi.fn().mockResolvedValue(undefined),
  exitFullscreen: vi.fn().mockResolvedValue(undefined),
}))

const success: CalibrationImagePrediction = {
  ok: true, prediction: { x: 0.5, y: 0.5 }, image: { width: 640, height: 480 },
  intrinsics: { focalLengthPx: 640, principalPoint: { x: 320, y: 240 } },
}

beforeEach(() => {
  vi.resetAllMocks()
  vi.mocked(loadCalibration).mockResolvedValue({ status: 'empty' })
  vi.mocked(saveCalibration).mockResolvedValue(undefined)
  const track = { label: 'Camera A', getSettings: () => ({ deviceId: 'camera-a', width: 640, height: 480, facingMode: 'user' }) }
  vi.mocked(useCamera).mockReturnValue({
    status: 'ready', error: null, stream: { getVideoTracks: () => [track] } as unknown as MediaStream,
    start: vi.fn().mockResolvedValue(undefined), stop: vi.fn(),
  })
  vi.mocked(captureBurst).mockImplementation(async () =>
    Array.from({ length: 3 }, () => new Blob(['jpeg'], { type: 'image/jpeg' })),
  )
  vi.mocked(predictCalibrationImage).mockImplementation(async () => ({
    ...success, prediction: CALIBRATION_POINTS[Math.floor((vi.mocked(predictCalibrationImage).mock.calls.length - 1) / 3)],
  } as CalibrationImagePrediction))
  vi.spyOn(HTMLMediaElement.prototype, 'play').mockResolvedValue(undefined)
  vi.spyOn(HTMLMediaElement.prototype, 'pause').mockImplementation(() => {})
})

test('nine capture bursts produce a valid calibration despite one rejected frame', async () => {
  let finish!: (value: CalibrationImagePrediction) => void
  vi.mocked(predictCalibrationImage).mockReturnValueOnce(new Promise((resolve) => { finish = resolve }))
  render(<App />)
  fireEvent.click(screen.getByRole('button', { name: /^Calibration/ }))
  fireEvent.click(screen.getByRole('button', { name: 'Begin' }))
  for (let i = 0; i < 9; i++) {
    expect(screen.getByText(`Point ${i + 1} of 9`)).toBeInTheDocument()
    await act(async () => { fireEvent.keyDown(window, { key: ' ', code: 'Space' }) })
  }
  expect(screen.getByRole('status')).toHaveTextContent('Processing calibration: 0 of 27')
  await act(async () => finish({ ok: false, stage: 'preprocess', reason: 'face_not_detected' }))
  expect(await screen.findByText(/Calibration ready. 26 of 27 frames used/)).toBeInTheDocument()
  expect(screen.getByText(/26 valid frames/)).toBeInTheDocument()
  expect(captureBurst).toHaveBeenCalledTimes(9)
  expect(predictCalibrationImage).toHaveBeenCalledTimes(27)
})
