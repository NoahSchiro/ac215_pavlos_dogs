import type { ScreenFraction } from '../vision/model'
import './gazeCrosshair.css'

interface Props {
  /** Where the model thinks the user is looking, as screen fractions. Null hides the crosshair. */
  gaze: ScreenFraction | null
  /** Optional unsmoothed prediction, drawn as a faint dot for debugging. */
  raw?: ScreenFraction | null
}

function toPosition(point: ScreenFraction | null | undefined) {
  if (!point || !Number.isFinite(point.x) || !Number.isFinite(point.y)) return null
  const clamp = (v: number) => Math.min(1, Math.max(0, v))
  return { left: `${clamp(point.x) * 100}%`, top: `${clamp(point.y) * 100}%` }
}

/** Overlay marking the model's gaze estimate on screen. */
export function GazeCrosshair({ gaze, raw = null }: Props) {
  const gazePosition = toPosition(gaze)
  const rawPosition = toPosition(raw)

  return (
    <div className="gaze-overlay" aria-hidden="true">
      {rawPosition && (
        <span className="gaze-raw" data-testid="gaze-raw" style={rawPosition} />
      )}
      {gazePosition && (
        <div className="gaze-crosshair" data-testid="gaze-crosshair" style={gazePosition}>
          <span className="gaze-crosshair-ring" />
          <span className="gaze-crosshair-line horizontal" />
          <span className="gaze-crosshair-line vertical" />
        </div>
      )}
    </div>
  )
}
