import { act, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import App from './App'
import { CALIBRATION_POINTS, type CalibrationResult } from './calibration/points'
import { predictCalibrationImage, type CalibrationImagePrediction } from './calibration/inference'
import { loadCalibration, saveCalibration, clearCalibration } from './calibration/storage'
import { createCalibrationRecord } from './calibration/record'
import { fitCalibration } from './calibration/ridge'
import { verifyCaptureCamera } from './calibration/cameraMetadata'
import { VALIDATION_POINTS } from './calibration/validation'
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
  targets?: readonly { x: number; y: number }[]
  purpose?: 'calibration' | 'validation'
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

function savedRecord(count = 9) {
  const capture = fakeResult(count)
  return createCalibrationRecord({
    capture, frames: capture.samples.map((sample) => ({ sample, result: successfulPrediction })),
    validFrameCounts: CALIBRATION_POINTS.map((_, i) => capture.samples.filter((sample) => sample.pointIndex === i).length),
    fit: fitCalibration(capture.samples.map(() => ({ x: 0.5, y: 0.5 })), capture.samples.map((sample) => sample.target)),
  })
}

function validationCapture(): CalibrationResult {
  return {
    completedAt: 20, camera: testSession().record.calibration.capture.camera,
    samples: VALIDATION_POINTS.flatMap((target, pointIndex) => Array.from({ length: 3 }, () => ({
      target, pointIndex, image: new Blob(['fresh-jpeg'], { type: 'image/jpeg' }),
      timestamp: 10, viewportWidth: 800, viewportHeight: 600,
    }))),
  }
}

describe('App', () => {
  it('captures held-out validation targets, reports paired errors, and never saves or refits them', async () => {
    vi.mocked(loadCalibration).mockResolvedValueOnce({ status: 'restored', record: testSession().record })
    render(<App />)
    fireEvent.click(await screen.findByRole('button', { name: 'Verify saved camera' }))
    await screen.findByText(/Saved calibration camera verified/)
    fireEvent.click(screen.getByRole('button', { name: 'Validate calibration' }))
    expect(calibrationProps!.purpose).toBe('validation')
    expect(calibrationProps!.targets).toEqual(VALIDATION_POINTS)
    const capture = validationCapture()
    capture.samples.forEach(({ target }) => predict.mockResolvedValueOnce({
      ...successfulPrediction, prediction: { x: 1 - target.x, y: target.y },
    } as CalibrationImagePrediction))
    await act(async () => calibrationProps!.onComplete(capture))
    const output = await screen.findByRole('textbox', { name: 'Validation metrics JSON' })
    const report = JSON.parse((output as HTMLTextAreaElement).value)
    expect(report.rawMeanError).toBeCloseTo(0.5, 12)
    expect(report.correctedMeanError).toBeCloseTo(0, 12)
    expect(report.validFrameCount).toBe(12)
    expect(saveCalibration).not.toHaveBeenCalled()
    expect(screen.getByText(/9 valid frames/)).toBeInTheDocument()
  })

  it('does not enable validation before saved camera verification', async () => {
    vi.mocked(loadCalibration).mockResolvedValueOnce({ status: 'restored', record: testSession().record })
    render(<App />)
    await screen.findByText(/Saved calibration restored/)
    expect(screen.getByRole('button', { name: 'Validate calibration' })).toBeDisabled()
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

  it('shows processing progress and can cancel without publishing a late result', async () => {
    let finish!: (value: CalibrationImagePrediction) => void
    predict.mockReturnValueOnce(new Promise((resolve) => { finish = resolve }))
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: /^Calibration/ }))
    await act(async () => calibrationProps!.onComplete(fakeResult(27)))
    expect(screen.getByRole('status')).toHaveTextContent('Processing calibration: 0 of 27 frames')
    expect(screen.getByRole('progressbar')).toHaveAttribute('value', '0')
    fireEvent.click(screen.getByRole('button', { name: 'Cancel processing' }))
    await act(async () => finish(successfulPrediction))
    expect(screen.getByRole('status')).toHaveTextContent('Calibration processing cancelled')
    expect(screen.queryByText(/Calibration ready/)).not.toBeInTheDocument()
    expect(predict).toHaveBeenCalledTimes(1)
    expect(screen.getByRole('button', { name: /^Calibration/ })).toBeInTheDocument()
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

  it('keeps the verified calibration and saved record when replacement predictions lack variation', async () => {
    const previous = testSession().record
    vi.mocked(loadCalibration).mockResolvedValueOnce({ status: 'restored', record: previous })
    render(<App />)
    fireEvent.click(await screen.findByRole('button', { name: 'Verify saved camera' }))
    await screen.findByText(/Saved calibration camera verified/)
    fireEvent.click(screen.getByRole('button', { name: /^Recalibrate/ }))
    predict.mockResolvedValue(successfulPrediction)
    await act(async () => calibrationProps!.onComplete(fakeResult(27)))
    expect(await screen.findByRole('alert')).toHaveTextContent(/insufficient.*variation/i)
    expect(screen.getByRole('alert')).toHaveTextContent('Previous calibration kept')
    expect(screen.getByText(/9 valid frames/)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Validate calibration' })).toBeEnabled()
    expect(saveCalibration).not.toHaveBeenCalled()
    fireEvent.click(screen.getByRole('button', { name: 'Retry calibration' }))
    expect(screen.getByTestId('calibration-screen')).toBeInTheDocument()
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

  it.each(['resolve', 'reject'] as const)('ignores a superseded task that later %ss', async (settle) => {
    let finish!: (value: CalibrationImagePrediction) => void
    let fail!: (reason: Error) => void
    predict.mockReturnValueOnce(new Promise((resolve, reject) => { finish = resolve; fail = reject }))
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: /^Calibration/ }))
    await act(async () => calibrationProps!.onComplete(fakeResult(27)))
    fireEvent.click(screen.getByRole('button', { name: /^Calibration/ }))
    await act(async () => calibrationProps!.onComplete(fakeResult(9)))
    await screen.findByText(/Calibration ready. 9 of 9/)
    await act(async () => {
      if (settle === 'resolve') finish(successfulPrediction)
      else fail(new Error('late runtime failure'))
    })
    expect(screen.getByText(/Calibration ready. 9 of 9/)).toBeInTheDocument()
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
    expect(predict).toHaveBeenCalledTimes(10)
  })

  it('ignores a capture callback from a cancelled session', async () => {
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: /^Calibration/ }))
    const oldCapture = calibrationProps!
    await act(async () => oldCapture.onCancel())
    fireEvent.click(screen.getByRole('button', { name: /^Calibration/ }))
    await act(async () => oldCapture.onComplete(fakeResult(27)))
    expect(screen.getByTestId('calibration-screen')).toBeInTheDocument()
    expect(predict).not.toHaveBeenCalled()
  })

  it('stops processing after unmount', async () => {
    let finish!: (value: CalibrationImagePrediction) => void
    predict.mockReturnValueOnce(new Promise((resolve) => { finish = resolve }))
    const { unmount } = render(<App />)
    fireEvent.click(screen.getByRole('button', { name: /^Calibration/ }))
    await act(async () => calibrationProps!.onComplete(fakeResult(27)))
    unmount()
    await act(async () => finish(successfulPrediction))
    expect(predict).toHaveBeenCalledTimes(1)
  })

  it('cancels processing when opening the maze and ignores the late prediction', async () => {
    let finish!: (value: CalibrationImagePrediction) => void
    predict.mockReturnValueOnce(new Promise((resolve) => { finish = resolve }))
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: /^Calibration/ }))
    await act(async () => calibrationProps!.onComplete(fakeResult(27)))
    fireEvent.click(screen.getByRole('button', { name: /^Maze/ }))
    await act(async () => finish(successfulPrediction))
    fireEvent.click(screen.getByRole('button', { name: /Back to Home/ }))
    expect(screen.queryByText(/Calibration ready/)).not.toBeInTheDocument()
    expect(screen.queryByRole('progressbar')).not.toBeInTheDocument()
    expect(predict).toHaveBeenCalledTimes(1)
  })

  it('restores saved calibration without automatically opening the camera, then verifies it on request', async () => {
    const record = savedRecord()
    vi.mocked(loadCalibration).mockResolvedValueOnce({ status: 'restored', record })
    render(<App />)
    expect(await screen.findByText(/Saved calibration restored/)).toBeInTheDocument()
    expect(screen.getByText(/9 valid frames/)).toBeInTheDocument()
    expect(verifyCaptureCamera).not.toHaveBeenCalled()
    fireEvent.click(screen.getByRole('button', { name: 'Verify saved camera' }))
    expect(await screen.findByText(/Saved calibration camera verified/)).toBeInTheDocument()
    expect(verifyCaptureCamera).toHaveBeenCalledWith(record.calibration.capture.camera, expect.any(AbortSignal))
  })

  it('prompts for recalibration when the restored camera cannot be matched', async () => {
    vi.mocked(loadCalibration).mockResolvedValueOnce({ status: 'restored', record: savedRecord() })
    vi.mocked(verifyCaptureCamera).mockRejectedValueOnce(new Error('Camera changed. Recalibrate before reuse.'))
    render(<App />)
    fireEvent.click(await screen.findByRole('button', { name: 'Verify saved camera' }))
    expect(await screen.findByRole('alert')).toHaveTextContent(/Camera changed.*Recalibrate/)
    expect(screen.queryByText(/camera verified/)).not.toBeInTheDocument()
  })

  it('reports an incompatible or corrupt saved record without activating it', async () => {
    vi.mocked(loadCalibration).mockResolvedValueOnce({ status: 'invalid', reason: 'Model changed. Recalibrate.' })
    render(<App />)
    expect(await screen.findByRole('alert')).toHaveTextContent('Model changed. Recalibrate.')
    expect(screen.queryByRole('button', { name: /^Recalibrate/ })).not.toBeInTheDocument()
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

  it('clears stored and in-memory calibration, and preserves them if clear fails', async () => {
    vi.mocked(loadCalibration).mockResolvedValueOnce({ status: 'restored', record: savedRecord() })
    render(<App />)
    await screen.findByText(/9 valid frames/)
    vi.mocked(clearCalibration).mockRejectedValueOnce(new Error('Storage locked'))
    fireEvent.click(screen.getByRole('button', { name: 'Clear calibration' }))
    expect(await screen.findByRole('alert')).toHaveTextContent(/Could not clear.*Storage locked/)
    expect(screen.getByText(/9 valid frames/)).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Clear calibration' }))
    expect(await screen.findByText(/Calibration cleared/)).toBeInTheDocument()
    expect(screen.queryByText(/9 valid frames/)).not.toBeInTheDocument()
  })

  it.each(['resolve', 'reject'] as const)('ignores a late save that %ss after clear', async (settle) => {
    let finish!: () => void
    let fail!: (error: Error) => void
    vi.mocked(saveCalibration).mockReturnValueOnce(new Promise((resolve, reject) => { finish = resolve; fail = reject }))
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: /^Calibration/ }))
    await act(async () => calibrationProps!.onComplete(fakeResult(9)))
    const signal = vi.mocked(saveCalibration).mock.calls[0][1]!
    fireEvent.click(screen.getByRole('button', { name: 'Clear calibration' }))
    expect(await screen.findByText(/Calibration cleared/)).toBeInTheDocument()
    expect(signal.aborted).toBe(true)
    await act(async () => { if (settle === 'resolve') finish(); else fail(new Error('late save error')) })
    expect(screen.getByText(/Calibration cleared/)).toBeInTheDocument()
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
    expect(screen.queryByText(/valid frames/)).not.toBeInTheDocument()
  })

  it('ignores a startup restore that resolves after a new calibration', async () => {
    let finish!: (value: Awaited<ReturnType<typeof loadCalibration>>) => void
    vi.mocked(loadCalibration).mockReturnValueOnce(new Promise((resolve) => { finish = resolve }))
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: /^Calibration/ }))
    await act(async () => calibrationProps!.onComplete(fakeResult(9)))
    await screen.findByText(/Calibration ready. 9 of 9/)
    await act(async () => finish({ status: 'restored', record: savedRecord(27) }))
    expect(screen.getByText(/9 valid frames/)).toBeInTheDocument()
    expect(screen.queryByText(/27 valid frames/)).not.toBeInTheDocument()
  })
})
