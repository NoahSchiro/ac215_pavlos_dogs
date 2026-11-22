import { render, screen } from '@testing-library/react'
import { expect, test } from 'vitest'
import { ErrorReport } from './ErrorReport'
import { suggestDrive } from './errorStats'
import type { ValidationSummary } from './validation'
import { MAZE_ROWS, mazeCellFraction } from '../config'

const viewport = { width: 1440, height: 900 }
const summary: ValidationSummary = {
  rawMeanError: 0.2,
  correctedMeanError: 0.1,
  validFrameCount: 12,
  viewport,
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
const cell = mazeCellFraction(viewport)

test('reports per-axis lean and spread in screen-height units, pixels and cells', () => {
  render(<ErrorReport summary={summary} />)
  const vertical = screen.getByRole('row', { name: /Vertical/ })
  expect(vertical).toHaveTextContent('5.0% low')
  expect(vertical).toHaveTextContent(`10.0% · 90 px · ${(0.1 / cell).toFixed(1)} cells`)
  expect(screen.getByRole('row', { name: /Horizontal/ })).toHaveTextContent('1.0% left')
})

test('names the lean at each target', () => {
  render(<ErrorReport summary={summary} />)
  expect(screen.getByText(/Top left: 1.0% right, 9.0% low/)).toBeInTheDocument()
  expect(screen.getByText(/Bottom right: 2.0% left, 1.0% high/)).toBeInTheDocument()
})

test('suggests a drive radius and a cell size for the measured error', () => {
  render(<ErrorReport summary={summary} />)
  const { radius } = suggestDrive(summary.corrected, cell)
  expect(screen.getByText(`GAZE_DRIVE_RADIUS = { x: ${radius.x}, y: ${radius.y} }`)).toBeInTheDocument()
  // Cells must be 2 x 10% = 20% of the screen height to cover the p90 vertical miss.
  const rows = Math.floor((cell * MAZE_ROWS) / 0.2)
  expect(screen.getByText(new RegExp(
    `at least 20.0% of the\\s+screen height \\(now ${(cell * 100).toFixed(1)}%\\): about ${rows} rows`,
  ))).toBeInTheDocument()
})
