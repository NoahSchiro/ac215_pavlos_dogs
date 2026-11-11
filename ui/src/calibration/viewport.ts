import type { CalibrationResult } from './points'
import type { Viewport } from '../vision/viewport'

/** Reuse existing sample metadata; mixed layouts cannot define one correction. */
export function captureViewport(capture: CalibrationResult): Viewport | null {
  const first = capture.samples[0]
  if (!first || !Number.isSafeInteger(first.viewportWidth) || first.viewportWidth <= 0 ||
      !Number.isSafeInteger(first.viewportHeight) || first.viewportHeight <= 0) return null
  return capture.samples.every((sample) => sample?.viewportWidth === first.viewportWidth && sample.viewportHeight === first.viewportHeight)
    ? { width: first.viewportWidth, height: first.viewportHeight } : null
}
