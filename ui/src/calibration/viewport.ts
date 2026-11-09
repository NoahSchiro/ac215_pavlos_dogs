import type { CalibrationResult } from './points'

export interface Viewport { width: number; height: number }

export function currentViewport(): Viewport {
  return { width: window.innerWidth, height: window.innerHeight }
}

export function sameViewport(expected: Viewport | null, actual: Viewport | null): boolean {
  return !!expected && !!actual && expected.width === actual.width && expected.height === actual.height
}

/** Reuse existing sample metadata; mixed layouts cannot define one correction. */
export function captureViewport(capture: CalibrationResult): Viewport | null {
  const first = capture.samples[0]
  if (!first || !Number.isSafeInteger(first.viewportWidth) || first.viewportWidth <= 0 ||
      !Number.isSafeInteger(first.viewportHeight) || first.viewportHeight <= 0) return null
  return capture.samples.every((sample) => sample?.viewportWidth === first.viewportWidth && sample.viewportHeight === first.viewportHeight)
    ? { width: first.viewportWidth, height: first.viewportHeight } : null
}
