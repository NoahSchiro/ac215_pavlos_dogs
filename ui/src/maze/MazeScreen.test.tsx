import { act, cleanup, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, expect, test, vi } from 'vitest'
import { MazeScreen } from './MazeScreen'
import { useCamera } from '../calibration/useCamera'
import { TEST_CAMERA, testSession } from '../calibration/fixtures/testSession'
import { getLandmarker } from '../vision/utils'
import { loadGazeModel, predictGaze } from '../vision/model'

vi.mock('../calibration/useCamera', () => ({ useCamera: vi.fn() }))
vi.mock('../vision/utils', () => ({ getLandmarker: vi.fn() }))
vi.mock('../vision/model', () => ({ loadGazeModel: vi.fn(), predictGaze: vi.fn() }))
vi.mock('./gazeCapture', () => ({ captureGazeFrame: vi.fn(() => ({ rows: 480, cols: 640, data: new Uint8Array(640 * 480 * 3) })) }))
vi.mock('./Maze', () => ({ default: ({ gazeDirection }: { gazeDirection?: string | null }) => <div data-testid="direction">{gazeDirection ?? 'stopped'}</div> }))

beforeEach(() => {
  vi.useFakeTimers()
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
})
afterEach(() => { cleanup(); vi.useRealTimers(); vi.restoreAllMocks() })

test('feeds a matching personal correction to maze movement', async () => {
  await act(async () => { render(<MazeScreen onBack={vi.fn()} calibration={testSession()} />) })
  expect(screen.getByTestId('direction')).toHaveTextContent('right')
  expect(predictGaze).toHaveBeenCalledWith(expect.anything(), expect.anything(), expect.anything(), {
    focalLengthPx: 640, principalPoint: { x: 320, y: 240 },
  })
})

test('falls back to raw movement without calibration', async () => {
  await act(async () => { render(<MazeScreen onBack={vi.fn()} />) })
  expect(screen.getByTestId('direction')).toHaveTextContent('left')
})

test.each(['face_not_detected', 'pose_failed'] as const)('stops movement on the next rejected frame: %s', async (reason) => {
  await act(async () => { render(<MazeScreen onBack={vi.fn()} calibration={testSession()} />) })
  expect(screen.getByTestId('direction')).toHaveTextContent('right')
  vi.mocked(predictGaze).mockResolvedValueOnce(reason)
  await act(async () => { await vi.advanceTimersByTimeAsync(200) })
  expect(screen.getByTestId('direction')).toHaveTextContent('stopped')
  expect(screen.getByRole('status')).toHaveTextContent(reason)
})

test('stops movement when a prediction throws', async () => {
  await act(async () => { render(<MazeScreen onBack={vi.fn()} calibration={testSession()} />) })
  expect(screen.getByTestId('direction')).toHaveTextContent('right')
  vi.mocked(predictGaze).mockRejectedValueOnce(new Error('inference failed'))
  await act(async () => { await vi.advanceTimersByTimeAsync(200) })
  expect(screen.getByTestId('direction')).toHaveTextContent('stopped')
})

test('does not run another prediction after an in-flight request finishes after unmount', async () => {
  let finish!: (point: { x: number; y: number }) => void
  vi.mocked(predictGaze).mockReturnValueOnce(new Promise((resolve) => { finish = resolve }))
  let unmount!: () => void
  await act(async () => { ({ unmount } = render(<MazeScreen onBack={vi.fn()} calibration={testSession()} />)) })
  unmount()
  await act(async () => { finish({ x: 0.2, y: 0.5 }); await vi.advanceTimersByTimeAsync(1000) })
  expect(predictGaze).toHaveBeenCalledTimes(1)
})
