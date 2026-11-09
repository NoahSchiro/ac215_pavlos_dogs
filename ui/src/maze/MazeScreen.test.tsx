import { act, cleanup, render, screen } from '@testing-library/react'
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
afterEach(() => { cleanup(); vi.useRealTimers(); vi.restoreAllMocks() })

test('feeds a matching personal correction to maze movement', async () => {
  await act(async () => { render(<MazeScreen onBack={vi.fn()} calibration={testSession()} />) })
  expect(screen.getByTestId('direction')).toHaveTextContent('right')
  expect(screen.getByTestId('gaze-crosshair')).toHaveStyle({ left: '80%', top: '50%' })
  expect(predictGaze).toHaveBeenCalledWith(expect.anything(), expect.anything(), expect.anything(), {
    focalLengthPx: 640, principalPoint: { x: 320, y: 240 },
  })
})

test.each(['face_not_detected', 'pose_failed', 'throw', 'non-finite', 'not-ready'] as const)('stops movement and hides the marker on an unusable frame: %s', async (reason) => {
  await act(async () => { render(<MazeScreen onBack={vi.fn()} calibration={testSession()} />) })
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
  await act(async () => { await vi.advanceTimersByTimeAsync(200) })
  const marker = screen.getByTestId('gaze-crosshair')
  expect(parseFloat(marker.style.left) / 100).toBeCloseTo(expected.x, 12)
  expect(parseFloat(marker.style.top) / 100).toBeCloseTo(expected.y, 12)
  // A corrected boundary measurement still moves right while the filter settles.
  expect(screen.getByTestId('direction')).toHaveTextContent(calibrated ? 'right' : 'left')
})

test('restarts smoothing when the active calibration changes', async () => {
  const { rerender } = await act(async () => render(<MazeScreen onBack={vi.fn()} calibration={testSession()} />))
  expect(screen.getByTestId('gaze-crosshair')).toHaveStyle({ left: '80%' })
  await act(async () => rerender(<MazeScreen onBack={vi.fn()} calibration={null} />))
  expect(screen.getByTestId('gaze-crosshair')).toHaveStyle({ left: '20%' })
  expect(screen.getByTestId('direction')).toHaveTextContent('left')
})

test('ignores an in-flight prediction after leaving the maze', async () => {
  let finish!: (point: { x: number; y: number }) => void
  vi.mocked(predictGaze).mockReturnValueOnce(new Promise((resolve) => { finish = resolve }))
  const { unmount } = await act(async () => render(<MazeScreen onBack={vi.fn()} calibration={testSession()} />))
  unmount()
  await act(async () => { finish({ x: 0.2, y: 0.5 }); await vi.advanceTimersByTimeAsync(200) })
  expect(predictGaze).toHaveBeenCalledTimes(1)
  expect(screen.queryByTestId('gaze-crosshair')).not.toBeInTheDocument()
})
