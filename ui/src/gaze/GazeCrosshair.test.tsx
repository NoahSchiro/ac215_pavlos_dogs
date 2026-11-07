import { describe, expect, it } from 'vitest'
import { render, screen } from '@testing-library/react'
import { GazeCrosshair } from './GazeCrosshair'

describe('GazeCrosshair', () => {
  it('places the crosshair at the gaze point', () => {
    render(<GazeCrosshair gaze={{ x: 0.25, y: 0.75 }} />)
    const crosshair = screen.getByTestId('gaze-crosshair')
    expect(crosshair.style.left).toBe('25%')
    expect(crosshair.style.top).toBe('75%')
  })

  it('hides the crosshair when there is no gaze estimate', () => {
    render(<GazeCrosshair gaze={null} />)
    expect(screen.queryByTestId('gaze-crosshair')).not.toBeInTheDocument()
  })

  it('keeps the crosshair on screen', () => {
    render(<GazeCrosshair gaze={{ x: 1.3, y: -0.2 }} />)
    const crosshair = screen.getByTestId('gaze-crosshair')
    expect(crosshair.style.left).toBe('100%')
    expect(crosshair.style.top).toBe('0%')
  })

  it('ignores non-finite predictions', () => {
    render(<GazeCrosshair gaze={{ x: Number.NaN, y: 0.5 }} />)
    expect(screen.queryByTestId('gaze-crosshair')).not.toBeInTheDocument()
  })

  it('draws the raw prediction only when given', () => {
    const { rerender } = render(<GazeCrosshair gaze={{ x: 0.5, y: 0.5 }} />)
    expect(screen.queryByTestId('gaze-raw')).not.toBeInTheDocument()

    rerender(<GazeCrosshair gaze={{ x: 0.5, y: 0.5 }} raw={{ x: 0.6, y: 0.4 }} />)
    const raw = screen.getByTestId('gaze-raw')
    expect(raw.style.left).toBe('60%')
    expect(raw.style.top).toBe('40%')
  })

  it('never blocks clicks or screen readers', () => {
    const { container } = render(<GazeCrosshair gaze={{ x: 0.5, y: 0.5 }} />)
    expect(container.firstElementChild).toHaveAttribute('aria-hidden', 'true')
  })
})
