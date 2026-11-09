import { act, fireEvent, render, screen } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { Calibration } from './Calibration'
import { CALIBRATION_POINTS, VALIDATION_POINTS, FRAMES_PER_POINT, type CalibrationResult } from './points'
import type { Camera, CameraStatus } from './useCamera'

vi.mock('./useCamera', () => ({ useCamera: vi.fn() }))
vi.mock('./capture', () => ({ captureBurst: vi.fn() }))
vi.mock('./fullscreen', () => ({ enterFullscreen: vi.fn().mockResolvedValue(undefined) }))

import { captureBurst } from './capture'
import { enterFullscreen } from './fullscreen'
import { useCamera } from './useCamera'

const mockedUseCamera = vi.mocked(useCamera)
const mockedBurst = vi.mocked(captureBurst)

let camera: Camera

function setCamera(status: CameraStatus, error: string | null = null) {
  camera = {
    status,
    stream: status === 'ready' ? ({} as MediaStream) : null,
    error,
    start: vi.fn().mockResolvedValue(undefined),
    stop: vi.fn(),
  }
  mockedUseCamera.mockReturnValue(camera)
}

function fakeFrames() {
  return Array.from(
    { length: FRAMES_PER_POINT },
    () => new Blob(['x'], { type: 'image/jpeg' }),
  )
}

async function pressSpace(repeat = false) {
  await act(async () => {
    fireEvent.keyDown(window, { key: ' ', code: 'Space', repeat })
  })
}

/** Render with the camera ready and click Begin so targets are showing. */
function renderStarted(onComplete = vi.fn(), onCancel = vi.fn()) {
  const utils = render(<Calibration onComplete={onComplete} onCancel={onCancel} />)
  fireEvent.click(screen.getByRole('button', { name: 'Begin' }))
  return { ...utils, onComplete, onCancel }
}

beforeEach(() => {
  vi.clearAllMocks()
  mockedBurst.mockImplementation(async () => fakeFrames())
  // jsdom does not implement media playback.
  HTMLMediaElement.prototype.play = vi.fn().mockResolvedValue(undefined)
  HTMLMediaElement.prototype.pause = vi.fn()
  setCamera('ready')
})

describe('Calibration', () => {
  it('captures only the four validation targets without changing the default nine-point grid', async () => {
    const onComplete = vi.fn()
    render(<Calibration mode="validation" onComplete={onComplete} onCancel={vi.fn()} />)
    fireEvent.click(screen.getByRole('button', { name: 'Begin' }))
    expect(screen.getByRole('application')).toHaveAttribute('aria-label', 'Gaze validation')
    for (let i = 0; i < 4; i++) await pressSpace()
    const result = onComplete.mock.calls[0][0] as CalibrationResult
    expect(result.samples).toHaveLength(12)
    result.samples.forEach((sample, i) => expect(sample.target).toEqual(VALIDATION_POINTS[Math.floor(i / 3)]))
    expect(CALIBRATION_POINTS).toHaveLength(9)
    expect(camera.stop).toHaveBeenCalledOnce()
  })

  it('starts the camera on mount', () => {
    render(<Calibration onComplete={vi.fn()} onCancel={vi.fn()} />)
    expect(camera.start).toHaveBeenCalledTimes(1)
  })

  it('waits on a Camera ready screen before showing targets', () => {
    render(<Calibration onComplete={vi.fn()} onCancel={vi.fn()} />)
    expect(screen.getByText('Camera ready')).toBeInTheDocument()
    expect(screen.queryByTestId('crosshair')).not.toBeInTheDocument()
    expect(enterFullscreen).not.toHaveBeenCalled()
  })

  it('ignores Space before the camera is ready', async () => {
    setCamera('requesting')
    render(<Calibration onComplete={vi.fn()} onCancel={vi.fn()} />)
    await pressSpace()
    expect(screen.queryByTestId('crosshair')).not.toBeInTheDocument()
    expect(enterFullscreen).not.toHaveBeenCalled()
  })

  it('enters full screen and shows the first target when Begin is clicked', () => {
    renderStarted()
    expect(enterFullscreen).toHaveBeenCalledTimes(1)
    expect(screen.getByText('Point 1 of 9')).toBeInTheDocument()
    expect(screen.getByTestId('crosshair')).toHaveStyle({ left: '10%', top: '10%' })
  })

  it('also begins on Space from the Camera ready screen without capturing', async () => {
    render(<Calibration onComplete={vi.fn()} onCancel={vi.fn()} />)
    await pressSpace()
    expect(enterFullscreen).toHaveBeenCalledTimes(1)
    expect(screen.getByText('Point 1 of 9')).toBeInTheDocument()
    expect(mockedBurst).not.toHaveBeenCalled()
  })

  it('Back on the Camera ready screen cancels and stops the camera', () => {
    const onCancel = vi.fn()
    render(<Calibration onComplete={vi.fn()} onCancel={onCancel} />)
    fireEvent.click(screen.getByRole('button', { name: 'Back' }))
    expect(camera.stop).toHaveBeenCalledTimes(1)
    expect(onCancel).toHaveBeenCalledTimes(1)
  })

  it('shows the first target at the top-left grid position', () => {
    renderStarted()
    expect(screen.getByText('Point 1 of 9')).toBeInTheDocument()
    const ch = screen.getByTestId('crosshair')
    expect(ch).toHaveStyle({ left: '10%', top: '10%' })
  })

  it('advances to the next target after a Space capture', async () => {
    renderStarted()
    await pressSpace()
    expect(mockedBurst).toHaveBeenCalledTimes(1)
    expect(screen.getByText('Point 2 of 9')).toBeInTheDocument()
    expect(screen.getByTestId('crosshair')).toHaveStyle({ left: '50%', top: '10%' })
  })

  it('ignores held-Space repeats but accepts the next fresh press', async () => {
    renderStarted()
    await pressSpace()
    await pressSpace(true)
    expect(mockedBurst).toHaveBeenCalledTimes(1)
    expect(screen.getByText('Point 2 of 9')).toBeInTheDocument()

    await pressSpace()
    expect(mockedBurst).toHaveBeenCalledTimes(2)
    expect(screen.getByText('Point 3 of 9')).toBeInTheDocument()
  })

  it.each([
    ['ready', 'Begin'],
    ['ready', 'Back'],
    ['denied', 'Retry'],
  ] as const)('preserves native Space activation for %s / %s', (status, name) => {
    setCamera(status)
    render(<Calibration onComplete={vi.fn()} onCancel={vi.fn()} />)
    const button = screen.getByRole('button', { name })
    button.focus()
    // jsdom does not synthesize native clicks; check that the shortcut leaves
    // the default keyboard action intact instead of beginning calibration.
    expect(fireEvent.keyDown(button, { key: ' ', code: 'Space' })).toBe(true)
    expect(enterFullscreen).not.toHaveBeenCalled()
    expect(mockedBurst).not.toHaveBeenCalled()
  })

  it('completes after nine captures with correctly labeled samples', async () => {
    const { onComplete } = renderStarted()

    for (let i = 0; i < 9; i++) await pressSpace()

    expect(onComplete).toHaveBeenCalledTimes(1)
    const result = onComplete.mock.calls[0][0]
    expect(result.samples).toHaveLength(9 * FRAMES_PER_POINT)
    expect(typeof result.completedAt).toBe('number')

    result.samples.forEach(
      (s: { target: unknown; pointIndex: number; image: Blob }, i: number) => {
        const expectedIndex = Math.floor(i / FRAMES_PER_POINT)
        expect(s.pointIndex).toBe(expectedIndex)
        expect(s.target).toEqual(CALIBRATION_POINTS[expectedIndex])
        expect(s.image).toBeInstanceOf(Blob)
      },
    )
    expect(camera.stop).toHaveBeenCalled()
  })

  it('does not capture a tenth point', async () => {
    const { onComplete } = renderStarted()
    for (let i = 0; i < 10; i++) await pressSpace()
    expect(onComplete).toHaveBeenCalledTimes(1)
    expect(mockedBurst).toHaveBeenCalledTimes(9)
  })

  it('ignores Space while the camera is still requesting', async () => {
    setCamera('requesting')
    render(<Calibration onComplete={vi.fn()} onCancel={vi.fn()} />)
    expect(screen.getByText('Starting camera')).toBeInTheDocument()
    await pressSpace()
    expect(mockedBurst).not.toHaveBeenCalled()
  })

  it('ignores Space while a burst is in progress', async () => {
    let finish!: (frames: Blob[]) => void
    mockedBurst.mockReturnValueOnce(new Promise((r) => (finish = r)))
    renderStarted()

    await pressSpace()
    expect(screen.getByText(/Capturing/)).toBeInTheDocument()
    await pressSpace()
    expect(mockedBurst).toHaveBeenCalledTimes(1)

    await act(async () => finish(fakeFrames()))
    expect(screen.getByText('Point 2 of 9')).toBeInTheDocument()
  })

  it('shows an error and stays on the same point when capture fails', async () => {
    mockedBurst.mockRejectedValueOnce(new Error('no frame'))
    renderStarted()
    await pressSpace()
    expect(screen.getByText(/Capture failed/)).toBeInTheDocument()
    expect(screen.getByText('Point 1 of 9')).toBeInTheDocument()

    await pressSpace()
    expect(screen.queryByText(/Capture failed/)).not.toBeInTheDocument()
    expect(screen.getByText('Point 2 of 9')).toBeInTheDocument()
  })

  it('cancels on Escape and stops the camera', async () => {
    const { onCancel } = renderStarted()
    await act(async () => {
      fireEvent.keyDown(window, { key: 'Escape' })
    })
    expect(camera.stop).toHaveBeenCalledTimes(1)
    expect(onCancel).toHaveBeenCalledTimes(1)
  })

  it.each(['cancel', 'unmount'] as const)(
    'discards a pending final burst after %s',
    async (action) => {
      const { onComplete, onCancel, unmount } = renderStarted()
      for (let i = 0; i < 8; i++) await pressSpace()
      let finish!: (frames: Blob[]) => void
      mockedBurst.mockReturnValueOnce(new Promise((r) => (finish = r)))
      await pressSpace()

      if (action === 'cancel') {
        fireEvent.keyDown(window, { key: 'Escape' })
        expect(onCancel).toHaveBeenCalledTimes(1)
      }
      unmount()
      await act(async () => finish(fakeFrames()))
      expect(onComplete).not.toHaveBeenCalled()
    },
  )

  it('does not show a late capture error after cancellation', async () => {
    let fail!: (reason: Error) => void
    mockedBurst.mockReturnValueOnce(new Promise((_, reject) => (fail = reject)))
    const { onComplete } = renderStarted()
    await pressSpace()
    fireEvent.keyDown(window, { key: 'Escape' })
    await act(async () => fail(new Error('camera stopped')))
    expect(screen.queryByText(/Capture failed/)).not.toBeInTheDocument()
    expect(screen.getByText('Point 1 of 9')).toBeInTheDocument()
    expect(onComplete).not.toHaveBeenCalled()
  })

  it('shows the camera error with Retry and Back when access is denied', async () => {
    setCamera('denied', 'Camera access was denied.')
    const onCancel = vi.fn()
    render(<Calibration onComplete={vi.fn()} onCancel={onCancel} />)

    expect(screen.getByRole('alert')).toHaveTextContent('Camera access was denied.')
    expect(screen.queryByTestId('crosshair')).not.toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: 'Retry' }))
    expect(camera.start).toHaveBeenCalledTimes(2) // mount + retry

    fireEvent.click(screen.getByRole('button', { name: 'Back' }))
    expect(onCancel).toHaveBeenCalledTimes(1)
  })

  it('removes the keyboard listener on unmount', async () => {
    const { unmount } = renderStarted()
    unmount()
    await pressSpace()
    expect(mockedBurst).not.toHaveBeenCalled()
  })
})
