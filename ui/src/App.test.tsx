import { act, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import App from './App'
import type { CalibrationResult } from './calibration/points'

type CalibrationProps = {
  onComplete: (r: CalibrationResult) => void
  onCancel: () => void
}
let calibrationProps: CalibrationProps | null = null

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
    samples: Array.from({ length: count }, (_, i) => ({
      image: new Blob(['x']),
      target: { x: 0.5, y: 0.5 },
      pointIndex: i,
      viewportWidth: 800,
      viewportHeight: 600,
      timestamp: 0,
    })),
    completedAt: 0,
  }
}

describe('App', () => {
  it('shows the home screen with Maze and Calibration options', () => {
    render(<App />)
    expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent('Gaze Maze')
    expect(screen.getByRole('button', { name: /^Maze/ })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /^Calibration/ })).toBeInTheDocument()
  })

  it('asks for a mode after choosing the maze, and can go back', () => {
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: /^Maze/ }))
    const modes = screen.getByRole('group', { name: 'Choose a mode' })
    expect(modes).toHaveTextContent('Single player')
    expect(modes).toHaveTextContent('Race a friend')

    fireEvent.click(screen.getByRole('button', { name: /Back/ }))
    expect(screen.getByRole('button', { name: /^Calibration/ })).toBeInTheDocument()
  })

  it('opens race mode from the maze menu', () => {
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: /^Maze/ }))
    fireEvent.click(screen.getByRole('button', { name: /^Race a friend/ }))
    expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent('Race a friend')
    expect(screen.getByRole('button', { name: 'Create race' })).toBeInTheDocument()
  })

  it('opens the maze and returns home', () => {
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: /^Maze/ }))
    fireEvent.click(screen.getByRole('button', { name: /^Single player/ }))
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

  it('returns home with the result after calibration completes', async () => {
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: /^Calibration/ }))
    fullscreenElement = document.documentElement

    await act(async () => calibrationProps!.onComplete(fakeResult(27)))

    expect(exitFullscreen).toHaveBeenCalledTimes(1)
    expect(screen.getByText(/27 frames captured/)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /^Recalibrate/ })).toBeInTheDocument()
  })

  it('returns home without a result when calibration is cancelled', async () => {
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: /^Calibration/ }))
    fullscreenElement = document.documentElement

    await act(async () => calibrationProps!.onCancel())

    expect(exitFullscreen).toHaveBeenCalledTimes(1)
    expect(screen.queryByText(/frames captured/)).not.toBeInTheDocument()
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
    expect(screen.getByText(/9 frames captured/)).toBeInTheDocument()
  })
})
