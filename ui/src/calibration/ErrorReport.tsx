import type { ValidationSummary } from './validation'
import { suggestDrive, type AxisError } from './errorStats'
import { GAZE_DRIVE_RADIUS, MAZE_ROWS, mazeCellFraction } from '../config'

const percent = (v: number) => `${(Math.abs(v) * 100).toFixed(1)}%`

/** Calibrated error per axis, per target, and what it implies for the maze. */
export function ErrorReport({ summary }: { summary: ValidationSummary }) {
  const { viewport, corrected } = summary
  const cell = mazeCellFraction(viewport)
  const suggestion = suggestDrive(corrected, cell)
  const size = (v: number) => `${percent(v)} · ${Math.round(Math.abs(v) * viewport.height)} px · ${(Math.abs(v) / cell).toFixed(1)} cells`
  const row = (label: string, error: AxisError, plus: string, minus: string) => (
    <tr>
      <th scope="row">{label}</th>
      <td>{percent(error.bias)} {error.bias >= 0 ? plus : minus}</td>
      <td>{size(error.meanAbs)}</td>
      <td>{size(error.p90Abs)}</td>
    </tr>
  )
  const rowsThatFit = suggestion.minCell > 0 ? Math.floor((cell * MAZE_ROWS) / suggestion.minCell) : MAZE_ROWS
  return (
    <div className="accuracy-report">
      <h2>Calibrated error by direction</h2>
      <table className="accuracy-table">
        <thead>
          <tr><th /><th scope="col">Leans</th><th scope="col">Typical miss</th><th scope="col">90% of misses within</th></tr>
        </thead>
        <tbody>
          {row('Horizontal', corrected.x, 'right', 'left')}
          {row('Vertical', corrected.y, 'low', 'high')}
        </tbody>
      </table>
      <h2>Lean at each target</h2>
      <ul>
        {summary.correctedByTarget.map(({ target, dx, dy, frames }) => (
          <li key={`${target.x},${target.y}`}>
            {target.y < 0.5 ? 'Top' : 'Bottom'} {target.x < 0.5 ? 'left' : 'right'}:{' '}
            {percent(dx)} {dx >= 0 ? 'right' : 'left'}, {percent(dy)} {dy >= 0 ? 'low' : 'high'} ({frames} frames)
          </li>
        ))}
      </ul>
      <h2>For the maze</h2>
      <p>
        Drive area for this error: <code>GAZE_DRIVE_RADIUS = {'{'} x: {suggestion.radius.x}, y: {suggestion.radius.y} {'}'}</code>{' '}
        (now <code>{'{'} x: {GAZE_DRIVE_RADIUS.x}, y: {GAZE_DRIVE_RADIUS.y} {'}'}</code>).
      </p>
      <p>
        To stay still while you look at Pavlos, a cell should be at least {percent(suggestion.minCell)} of the
        screen height (now {percent(cell)}): about {rowsThatFit} rows on this screen instead of {MAZE_ROWS}.
      </p>
      <p>
        Percentages are of the screen height on both axes. These are single frames without smoothing, so the miss in
        play is somewhat smaller; with {corrected.frames} frames, run the check a few times before changing settings.
      </p>
    </div>
  )
}
