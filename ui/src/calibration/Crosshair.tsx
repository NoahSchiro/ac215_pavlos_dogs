import type { NormalizedPoint } from './points'

interface Props {
  point: NormalizedPoint
  /** Frames are being captured: the marker turns amber. */
  capturing?: boolean
}

/** A target marker positioned by normalized viewport coordinates. */
export function Crosshair({ point, capturing = false }: Props) {
  return (
    <div
      className={capturing ? 'crosshair capturing' : 'crosshair'}
      data-testid="crosshair"
      aria-hidden="true"
      style={{ left: `${point.x * 100}%`, top: `${point.y * 100}%` }}
    >
      <span className="crosshair-ring" />
      <span className="crosshair-line horizontal" />
      <span className="crosshair-line vertical" />
    </div>
  )
}
