import { act, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import App from './App'
import { CALIBRATION_POINTS, VALIDATION_POINTS, type CalibrationResult } from './calibration/points'
import { predictCalibrationImage, type CalibrationImagePrediction } from './calibration/inference'
import { loadCalibration, saveCalibration, clearCalibration } from './calibration/storage'
import { verifyCaptureCamera } from './calibration/cameraMetadata'
import { testSession } from './calibration/fixtures/testSession'

vi.mock('./calibration/storage', () => ({ loadCalibration: vi.fn(), saveCalibration: vi.fn(), clearCalibration: vi.fn() }))
vi.mock('./calibration/cameraMetadata', async (importOriginal) => ({
  ...await importOriginal<typeof import('./calibration/cameraMetadata')>(), verifyCaptureCamera: vi.fn(),
}))

// Keep the real processing and ridge functions; mock the browser/WASM boundary.
vi.mock('./calibration/inference', () => ({ predictCalibrationImage: vi.fn() }))
const predict = vi.mocked(predictCalibrationImage)
const successfulPrediction: CalibrationImagePrediction = {
  ok: true, prediction: { x: 0.5, y: 0.5 }, image: { width: 640, height: 480 },
  intrinsics: { focalLengthPx: 640, principalPoint: { x: 320, y: 240 } },
}

type CalibrationProps = {
  onComplete: (r: CalibrationResult) => void
  onCancel: () => void
  mode?: 'calibration' | 'validation'
}
let calibrationProps: CalibrationProps | null = null
const targetByImage = new WeakMap<Blob, { x: number; y: number }>()

vi.mock('./calibration/Calibration', () => ({
  Calibration: (props: CalibrationProps) => {
    calibrationProps = props
    return <div data-testid="calibration-screen" />
  },
}))

const requestFullscreen = vi.fn().mockResolvedValue(undefined)
const exitFullscreen = vi.fn().mockResolvedValue(undefined)
let fullscreenElement: Element | null = null

beforeEach(() => {
  vi.mocked(loadCalibration).mockReset().mockResolvedValue({ status: 'empty' })
  vi.mocked(saveCalibration).mockReset().mockResolvedValue(undefined)
  vi.mocked(clearCalibration).mockReset().mockResolvedValue(undefined)
  vi.mocked(verifyCaptureCamera).mockReset().mockResolvedValue(undefined)
  predict.mockReset().mockImplementation(async (image) => ({
    ...successfulPrediction, prediction: targetByImage.get(image) ?? { x: 0.5, y: 0.5 },
  } as CalibrationImagePrediction))
  calibrationProps = null
  fullscreenElement = null
  requestFullscreen.mockClear()
  exitFullscreen.mockClear()
  document.documentElement.requestFullscreen = requestFullscreen
  document.exitFullscreen = exitFullscreen
  Object.defineProperty(document, 'fullscreenElement', {
    configurable: true,
    get: () => fullscreenElement,
  })
})

afterEach(() => {
  vi.restoreAllMocks()
})

function fakeResult(count: number): CalibrationResult {
  return {
    samples: Array.from({ length: count }, (_, i) => {
      const image = new Blob(['x'], { type: 'image/jpeg' })
      targetByImage.set(image, CALIBRATION_POINTS[i % 9])
      return {
        image,
        target: CALIBRATION_POINTS[i % 9],
        pointIndex: i % 9,
        viewportWidth: 800,
        viewportHeight: 600,
        timestamp: 0,
      }
    }),
    completedAt: 0,
    camera: { deviceId: 'camera-a', label: 'Camera A', width: 640, height: 480, facingMode: 'user' },
  }
}

function validationCapture(): CalibrationResult {
  const samples = VALIDATION_POINTS.flatMap((target, pointIndex) => Array.from({ length: 3 }, () => {
    const image = new Blob(['fresh'], { type: 'image/jpeg' })
    targetByImage.set(image, { x: 1 - target.x, y: target.y })
    return { image, target, pointIndex, timestamp: 10, viewportWidth: 800, viewportHeight: 600 }
  }))
  return { samples, completedAt: 20, camera: testSession().record.calibration.capture.camera }
}

describe('App', () => {
  it('opens a simple four-point check and displays paired errors without saving another fit', async () => {
    vi.mocked(loadCalibration).mockResolvedValueOnce({ status: 'restored', record: testSession().record })
    render(<App />)
    expect(await screen.findByRole('button', { name: 'Verify saved camera' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Validate calibration' })).toBeDisabled()
    fireEvent.click(screen.getByRole('button', { name: 'Verify saved camera' }))
    await screen.findByText(/Saved calibration camera verified/)
    fireEvent.click(screen.getByRole('button', { name: 'Validate calibration' }))
    expect(calibrationProps!.mode).toBe('validation')
    await act(async () => calibrationProps!.onComplete(validationCapture()))
    expect(await screen.findByText('Raw mean error: 0.500000')).toBeInTheDocument()
    expect(screen.getByText('Corrected mean error: 0.000000')).toBeInTheDocument()
    expect(saveCalibration).not.toHaveBeenCalled()
    fireEvent.click(screen.getByRole('button', { name: 'Back to Home' }))
    expect(screen.getByText(/9 valid frames/)).toBeInTheDocument()
  })

  it('stops checking when returning home and ignores the pending prediction', async () => {
    vi.mocked(loadCalibration).mockResolvedValueOnce({ status: 'restored', record: testSession().record })
    render(<App />)
    fireEvent.click(await screen.findByRole('button', { name: 'Verify saved camera' }))
    await screen.findByText(/Saved calibration camera verified/)
    fireEvent.click(screen.getByRole('button', { name: 'Validate calibration' }))
    let resolve!: (result: CalibrationImagePrediction) => void
    predict.mockImplementationOnce(() => new Promise((done) => { resolve = done }))
    await act(async () => calibrationProps!.onComplete(validationCapture()))
    expect(screen.getByRole('status')).toHaveTextContent('Checking accuracy: 0 of 12 frames.')
    fireEvent.click(screen.getByRole('button', { name: 'Back to Home' }))
    await act(async () => resolve(successfulPrediction))
    expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent('Gaze Maze')
    expect(screen.queryByText(/Raw mean error/)).not.toBeInTheDocument()
    expect(predict).toHaveBeenCalledTimes(1)
    expect(saveCalibration).not.toHaveBeenCalled()
  })

  it('shows the home screen with Maze and Calibration options', () => {
    render(<App />)
    expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent('Gaze Maze')
    expect(screen.getByRole('button', { name: /^Maze/ })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /^Calibration/ })).toBeInTheDocument()
  })

  it('opens the maze and returns home', () => {
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: /^Maze/ }))
    expect(screen.getByRole('group', { name: 'Maze' })).toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: /Back to Home/i }))
    expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent('Gaze Maze')
  })

  it('opens the calibration screen without requesting full screen itself', () => {
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: /^Calibration/ }))
    expect(screen.getByTestId('calibration-screen')).toBeInTheDocument()
    // Full screen is entered by the calibration screen after the camera
    // permission prompt, so the prompt is never hidden (Safari).
    expect(requestFullscreen).not.toHaveBeenCalled()
  })

  it('returns home and fits the capture before declaring calibration ready', async () => {
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: /^Calibration/ }))
    fullscreenElement = document.documentElement

    await act(async () => calibrationProps!.onComplete(fakeResult(27)))

    expect(exitFullscreen).toHaveBeenCalledTimes(1)
    expect(await screen.findByText(/Calibration ready. 27 of 27 frames used/)).toBeInTheDocument()
    expect(predict).toHaveBeenCalledTimes(27)
    expect(screen.getByRole('button', { name: /^Recalibrate/ })).toBeInTheDocument()
  })

  it('returns home without a result when calibration is cancelled', async () => {
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: /^Calibration/ }))
    fullscreenElement = document.documentElement

    await act(async () => calibrationProps!.onCancel())

    expect(exitFullscreen).toHaveBeenCalledTimes(1)
    expect(screen.queryByText(/Calibration ready/)).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: /^Calibration/ })).toBeInTheDocument()
  })

  it('does not call exitFullscreen when not in full screen', async () => {
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: /^Calibration/ }))
    await act(async () => calibrationProps!.onCancel())
    expect(exitFullscreen).not.toHaveBeenCalled()
  })

  it('replaces the previous result on recalibration', async () => {
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: /^Calibration/ }))
    await act(async () => calibrationProps!.onComplete(fakeResult(27)))

    fireEvent.click(screen.getByRole('button', { name: /^Recalibrate/ }))
    await act(async () => calibrationProps!.onComplete(fakeResult(9)))
    expect(await screen.findByText(/Calibration ready. 9 of 9 frames used/)).toBeInTheDocument()
  })

  it('preserves the previous calibration when a new capture has insufficient coverage and offers retry', async () => {
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: /^Calibration/ }))
    await act(async () => calibrationProps!.onComplete(fakeResult(27)))
    await screen.findByText(/Calibration ready/)
    fireEvent.click(screen.getByRole('button', { name: /^Recalibrate/ }))
    await act(async () => calibrationProps!.onComplete(fakeResult(8)))
    expect(await screen.findByRole('alert')).toHaveTextContent(/target.*9/i)
    expect(screen.getByRole('alert')).toHaveTextContent('Previous calibration kept')
    expect(screen.getByText(/27 valid frames/)).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Retry calibration' }))
    expect(screen.getByTestId('calibration-screen')).toBeInTheDocument()
    await act(async () => calibrationProps!.onComplete(fakeResult(9)))
    expect(await screen.findByText(/Calibration ready. 9 of 9 frames used/)).toBeInTheDocument()
  })

  it('preserves an existing calibration when processing is cancelled', async () => {
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: /^Calibration/ }))
    await act(async () => calibrationProps!.onComplete(fakeResult(27)))
    await screen.findByText(/Calibration ready/)
    let finish!: (value: CalibrationImagePrediction) => void
    predict.mockReturnValueOnce(new Promise((resolve) => { finish = resolve }))
    fireEvent.click(screen.getByRole('button', { name: /^Recalibrate/ }))
    await act(async () => calibrationProps!.onComplete(fakeResult(9)))
    fireEvent.click(screen.getByRole('button', { name: 'Cancel processing' }))
    await act(async () => finish(successfulPrediction))
    expect(screen.getByText(/27 valid frames/)).toBeInTheDocument()
    expect(screen.queryByText(/Calibration ready. 9/)).not.toBeInTheDocument()
  })

  it('publishes a new calibration only after its save commits', async () => {
    let commit!: () => void
    vi.mocked(saveCalibration).mockReturnValueOnce(new Promise((resolve) => { commit = resolve }))
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: /^Calibration/ }))
    await act(async () => calibrationProps!.onComplete(fakeResult(9)))
    expect(screen.getByRole('status')).toHaveTextContent('Saving calibration')
    expect(screen.queryByText(/Calibration ready/)).not.toBeInTheDocument()
    await act(async () => commit())
    expect(await screen.findByText(/Calibration ready. 9 of 9/)).toBeInTheDocument()
  })

  it('preserves the previous calibration when saving a replacement fails', async () => {
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: /^Calibration/ }))
    await act(async () => calibrationProps!.onComplete(fakeResult(27)))
    await screen.findByText(/Calibration ready/)
    vi.mocked(saveCalibration).mockRejectedValueOnce(new DOMException('Quota exceeded', 'QuotaExceededError'))
    fireEvent.click(screen.getByRole('button', { name: /^Recalibrate/ }))
    await act(async () => calibrationProps!.onComplete(fakeResult(9)))
    expect(await screen.findByRole('alert')).toHaveTextContent(/not saved.*Quota exceeded.*Previous calibration kept/)
    expect(screen.getByText(/27 valid frames/)).toBeInTheDocument()
  })
})
