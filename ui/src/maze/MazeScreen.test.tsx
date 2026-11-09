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
  expect(predictGaze).toHaveBeenCalledWith(expect.anything(), expect.anything(), expect.anything(), {
    focalLengthPx: 640, principalPoint: { x: 320, y: 240 },
  })
})

test.each(['face_not_detected', 'pose_failed'] as const)('stops movement on the next rejected frame: %s', async (reason) => {
  await act(async () => { render(<MazeScreen onBack={vi.fn()} calibration={testSession()} />) })
  expect(screen.getByTestId('direction')).toHaveTextContent('right')
  vi.mocked(predictGaze).mockResolvedValueOnce(reason)
  await act(async () => { await vi.advanceTimersByTimeAsync(200) })
  expect(screen.getByTestId('direction')).toHaveTextContent('stopped')
  expect(screen.getByRole('status')).toHaveTextContent(reason)
})
