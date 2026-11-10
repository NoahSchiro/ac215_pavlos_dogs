import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, expect, test, vi } from 'vitest'
import { MazeScreen } from './MazeScreen'
import { useCamera } from '../calibration/useCamera'
import { TEST_CAMERA, testSession } from '../calibration/fixtures/testSession'
import { getLandmarker } from '../vision/utils'
import { loadGazeModel, predictGaze } from '../vision/model'
import { GazeKalmanFilter } from '../gaze/kalman'

vi.mock('../calibration/useCamera', () => ({ useCamera: vi.fn() }))
vi.mock('../vision/utils', () => ({ getLandmarker: vi.fn() }))
vi.mock('../vision/model', () => ({ loadGazeModel: vi.fn(), predictGaze: vi.fn() }))
vi.mock('./Maze', () => ({ default: ({ gazeDirection }: { gazeDirection?: string | null }) => <div data-testid="direction">{gazeDirection ?? 'stopped'}</div> }))

beforeEach(() => {
  vi.useFakeTimers()
  vi.spyOn(performance, 'now').mockImplementation(() => Date.now())
  vi.resetAllMocks()
  vi.stubGlobal('innerWidth', 800)
  vi.stubGlobal('innerHeight', 600)
  Object.defineProperty(document, 'fullscreenElement', { configurable: true, value: null })
  document.documentElement.requestFullscreen = vi.fn().mockResolvedValue(undefined)
  vi.mocked(useCamera).mockReturnValue({
    start: vi.fn().mockResolvedValue(undefined), stop: vi.fn(), status: 'ready', error: null,
    stream: { getVideoTracks: () => [{ label: 'Camera A', getSettings: () => TEST_CAMERA }] } as unknown as MediaStream,
  })
  vi.mocked(getLandmarker).mockResolvedValue({} as Awaited<ReturnType<typeof getLandmarker>>)
  vi.mocked(loadGazeModel).mockResolvedValue({} as Awaited<ReturnType<typeof loadGazeModel>>)
  vi.mocked(predictGaze).mockResolvedValue({ x: 0.2, y: 0.5 })
  vi.spyOn(HTMLMediaElement.prototype, 'play').mockResolvedValue(undefined)
  vi.spyOn(HTMLMediaElement.prototype, 'pause').mockImplementation(() => {})
  vi.spyOn(HTMLMediaElement.prototype, 'readyState', 'get').mockReturnValue(2)
  vi.spyOn(HTMLVideoElement.prototype, 'videoWidth', 'get').mockReturnValue(640)
  vi.spyOn(HTMLVideoElement.prototype, 'videoHeight', 'get').mockReturnValue(480)
  vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue({
    drawImage: vi.fn(), getImageData: () => ({ data: new Uint8ClampedArray(640 * 480 * 4) }),
  } as unknown as CanvasRenderingContext2D)
})
afterEach(() => { cleanup(); vi.useRealTimers(); vi.restoreAllMocks(); vi.unstubAllGlobals() })

async function startGame() {
  await act(async () => { fireEvent.click(screen.getByRole('button', { name: 'Start game' })) })
}

test.each([true, false])('waits for camera permission, then starts after the fullscreen click (calibrated: %s)', async (calibrated) => {
  const readyCamera = vi.mocked(useCamera).getMockImplementation()!()
  vi.mocked(useCamera).mockReturnValue({ ...readyCamera, status: 'requesting', stream: null })
  const props = { onBack: vi.fn(), calibration: calibrated ? testSession() : null }
  const { rerender } = render(<MazeScreen {...props} />)
  expect(screen.getByRole('button', { name: 'Start game' })).toBeDisabled()
  expect(document.documentElement.requestFullscreen).not.toHaveBeenCalled()
  expect(predictGaze).not.toHaveBeenCalled()

  // The permission prompt changed the viewport before the camera became ready.
  vi.stubGlobal('innerWidth', 1200)
  fireEvent(window, new Event('resize'))
  vi.mocked(useCamera).mockReturnValue(readyCamera)
  rerender(<MazeScreen {...props} />)
  let finish!: () => void
  vi.mocked(document.documentElement.requestFullscreen).mockImplementationOnce(() => new Promise<void>((resolve) => {
    finish = () => {
      Object.defineProperty(document, 'fullscreenElement', { configurable: true, value: document.documentElement })
      vi.stubGlobal('innerWidth', 800)
      document.dispatchEvent(new Event('fullscreenchange'))
      resolve()
    }
  }))
  fireEvent.click(screen.getByRole('button', { name: 'Start game' }))
  expect(document.documentElement.requestFullscreen).toHaveBeenCalledOnce()
  expect(predictGaze).not.toHaveBeenCalled()
  await act(async () => finish())
  expect(screen.getByTestId('gaze-crosshair')).toHaveStyle({ left: calibrated ? '80%' : '20%' })
  expect(screen.getByTestId('direction')).toHaveTextContent(calibrated ? 'right' : 'left')
  expect(screen.queryByRole('button', { name: 'Restore full screen' })).not.toBeInTheDocument()
})

test('pauses immediately on resize and resumes when the calibration viewport is restored', async () => {
  await act(async () => { render(<MazeScreen onBack={vi.fn()} calibration={testSession()} />) })
  await startGame()
  vi.stubGlobal('innerWidth', 1200)
  fireEvent(window, new Event('resize'))
  expect(screen.getByTestId('direction')).toHaveTextContent('stopped')
  expect(screen.queryByTestId('gaze-crosshair')).not.toBeInTheDocument()
  expect(screen.getByRole('status')).toHaveTextContent(/window|viewport/i)
  vi.stubGlobal('innerWidth', 800)
  await act(async () => { fireEvent(window, new Event('resize')) })
  expect(screen.getByTestId('direction')).toHaveTextContent('right')
})

test('does not use a saved fit when the starting viewport differs', async () => {
  vi.stubGlobal('innerWidth', 1200)
  await act(async () => { render(<MazeScreen onBack={vi.fn()} calibration={testSession()} />) })
  await startGame()
  expect(predictGaze).not.toHaveBeenCalled()
  expect(screen.getByTestId('direction')).toHaveTextContent('stopped')
  expect(screen.getByRole('button', { name: 'Restore full screen' })).toBeInTheDocument()
})

test('recovers from a failed fullscreen entry, then pauses on exit even at the same size', async () => {
  let full = false
  Object.defineProperty(document, 'fullscreenElement', { configurable: true,
    get: () => full ? document.documentElement : null })
  const request = vi.fn().mockRejectedValueOnce(new Error('Fullscreen refused')).mockImplementation(async () => {
    full = true
    vi.stubGlobal('innerWidth', 800)
    document.dispatchEvent(new Event('fullscreenchange'))
  })
  document.documentElement.requestFullscreen = request
  vi.stubGlobal('innerWidth', 1200)
  await act(async () => { render(<MazeScreen onBack={vi.fn()} calibration={testSession()} />) })
  await startGame()
  await act(async () => fireEvent.click(screen.getByRole('button', { name: 'Restore full screen' })))
  expect(request).toHaveBeenCalledTimes(2)
  expect(screen.getByTestId('direction')).toHaveTextContent('right')
  full = false
  fireEvent(document, new Event('fullscreenchange'))
  expect(screen.getByTestId('direction')).toHaveTextContent('stopped')
  expect(screen.queryByTestId('gaze-crosshair')).not.toBeInTheDocument()
})

test('feeds a matching personal correction to maze movement', async () => {
  await act(async () => { render(<MazeScreen onBack={vi.fn()} calibration={testSession()} />) })
  await startGame()
  expect(screen.getByTestId('direction')).toHaveTextContent('right')
  expect(screen.getByTestId('gaze-crosshair')).toHaveStyle({ left: '80%', top: '50%' })
  expect(predictGaze).toHaveBeenCalledWith(expect.anything(), expect.anything(), expect.anything(), {
    focalLengthPx: 640, principalPoint: { x: 320, y: 240 },
  })
})

test.each(['face_not_detected', 'pose_failed', 'throw', 'non-finite', 'not-ready'] as const)('stops movement and hides the marker on an unusable frame: %s', async (reason) => {
  await act(async () => { render(<MazeScreen onBack={vi.fn()} calibration={testSession()} />) })
  await startGame()
  expect(screen.getByTestId('direction')).toHaveTextContent('right')
  if (reason === 'throw') vi.mocked(predictGaze).mockRejectedValueOnce(new Error('Inference failed'))
  else if (reason === 'non-finite') vi.mocked(predictGaze).mockResolvedValueOnce({ x: NaN, y: 0.5 })
  else if (reason === 'not-ready') vi.spyOn(HTMLMediaElement.prototype, 'readyState', 'get').mockReturnValue(0)
  else vi.mocked(predictGaze).mockResolvedValueOnce(reason)
  await act(async () => { await vi.advanceTimersByTimeAsync(200) })
  expect(screen.getByTestId('direction')).toHaveTextContent('stopped')
  expect(screen.queryByTestId('gaze-crosshair')).not.toBeInTheDocument()
})

test.each([true, false])('uses the same Kalman output for crosshair and movement (calibrated: %s)', async (calibrated) => {
  const filter = new GazeKalmanFilter()
  const first = { x: calibrated ? 0.8 : 0.2, y: 0.5 }
  const second = { x: calibrated ? 0.6 : 0.35, y: 0.5 }
  const time = performance.now()
  filter.update(first, time)
  const expected = filter.update(second, time + 200)!
  vi.mocked(predictGaze).mockResolvedValueOnce({ x: 0.2, y: 0.5 })
    .mockResolvedValueOnce({ x: calibrated ? 0.4 : 0.35, y: 0.5 })
  await act(async () => { render(<MazeScreen onBack={vi.fn()} calibration={calibrated ? testSession() : null} />) })
  await startGame()
  await act(async () => { await vi.advanceTimersByTimeAsync(200) })
  const marker = screen.getByTestId('gaze-crosshair')
  expect(parseFloat(marker.style.left) / 100).toBeCloseTo(expected.x, 12)
  expect(parseFloat(marker.style.top) / 100).toBeCloseTo(expected.y, 12)
  // A corrected boundary measurement still moves right while the filter settles.
  expect(screen.getByTestId('direction')).toHaveTextContent(calibrated ? 'right' : 'left')
})

test('restarts smoothing when the active calibration changes', async () => {
  const { rerender } = await act(async () => render(<MazeScreen onBack={vi.fn()} calibration={testSession()} />))
  await startGame()
  expect(screen.getByTestId('gaze-crosshair')).toHaveStyle({ left: '80%' })
  await act(async () => rerender(<MazeScreen onBack={vi.fn()} calibration={null} />))
  expect(screen.getByTestId('gaze-crosshair')).toHaveStyle({ left: '20%' })
  expect(screen.getByTestId('direction')).toHaveTextContent('left')
})

test.each(['leave', 'resize'] as const)('ignores an in-flight prediction after %s', async (action) => {
  let finish!: (point: { x: number; y: number }) => void
  vi.mocked(predictGaze).mockReturnValueOnce(new Promise((resolve) => { finish = resolve }))
  const { unmount } = await act(async () => render(<MazeScreen onBack={vi.fn()} calibration={testSession()} />))
  await startGame()
  if (action === 'leave') unmount()
  else {
    vi.stubGlobal('innerWidth', 1200)
    fireEvent(window, new Event('resize'))
  }
  await act(async () => { finish({ x: 0.2, y: 0.5 }); await vi.advanceTimersByTimeAsync(200) })
  expect(predictGaze).toHaveBeenCalledTimes(1)
  expect(screen.queryByTestId('gaze-crosshair')).not.toBeInTheDocument()
})
