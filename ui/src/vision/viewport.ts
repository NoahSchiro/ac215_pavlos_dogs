export interface Viewport { width: number; height: number }

export function currentViewport(): Viewport {
  return { width: window.innerWidth, height: window.innerHeight }
}

export function sameViewport(expected: Viewport | null, actual: Viewport | null): boolean {
  return !!expected && !!actual && expected.width === actual.width && expected.height === actual.height
}
