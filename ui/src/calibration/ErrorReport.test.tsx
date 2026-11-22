import { render, screen } from '@testing-library/react'
import { expect, test } from 'vitest'
import { ErrorReport } from './ErrorReport'
import type { ValidationSummary } from './validation'

// 1440x900: the maze is 800 px tall, so one cell is 53 px (5.9% of the height).
const summary: ValidationSummary = {
  rawMeanError: 0.2,
  correctedMeanError: 0.1,
  validFrameCount: 12,
  viewport: { width: 1440, height: 900 },
  raw: { frames: 12, x: { bias: 0, meanAbs: 0.1, p90Abs: 0.2 }, y: { bias: 0, meanAbs: 0.1, p90Abs: 0.2 } },
  corrected: {
    frames: 12,
    x: { bias: -0.01, meanAbs: 0.03, p90Abs: 0.04 },
    y: { bias: 0.05, meanAbs: 0.07, p90Abs: 0.1 },
  },
  correctedByTarget: [
    { target: { x: 0.25, y: 0.25 }, frames: 3, dx: 0.01, dy: 0.09 },
    { target: { x: 0.75, y: 0.75 }, frames: 3, dx: -0.02, dy: -0.01 },
  ],
}

test('reports per-axis lean and spread in screen-height units, pixels and cells', () => {
  render(<ErrorReport summary={summary} />)
  const vertical = screen.getByRole('row', { name: /Vertical/ })
  expect(vertical).toHaveTextContent('5.0% low')
  expect(vertical).toHaveTextContent('10.0% · 90 px · 1.7 cells')
  expect(screen.getByRole('row', { name: /Horizontal/ })).toHaveTextContent('1.0% left')
})

test('names the lean at each target', () => {
  render(<ErrorReport summary={summary} />)
  expect(screen.getByText(/Top left: 1.0% right, 9.0% low/)).toBeInTheDocument()
  expect(screen.getByText(/Bottom right: 2.0% left, 1.0% high/)).toBeInTheDocument()
})

test('suggests a drive radius and a cell size for the measured error', () => {
  render(<ErrorReport summary={summary} />)
  // Two cells (0.119) plus the p90 miss, rounded up.
  expect(screen.getByText(/x: 0.16, y: 0.22/)).toBeInTheDocument()
  // Cells must be 2 x 10% = 20% of the height: 4 rows fit in an 89%-tall maze.
  expect(screen.getByText(/at least 20.0% of the\s+screen height \(now 5.9%\): about 4 rows/)).toBeInTheDocument()
})
