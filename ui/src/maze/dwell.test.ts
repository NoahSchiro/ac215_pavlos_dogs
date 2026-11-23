import { describe, expect, it } from 'vitest'
import { isDwelling, trackFixation, type Fixation } from './dwell'

const WIDE = { width: 1600, height: 800 }
const RADIUS = 0.08

function feed(points: Array<{ x: number; y: number } | null>, stepMs = 200) {
  let fixation: Fixation | null = null
  points.forEach((point, i) => { fixation = trackFixation(fixation, point, i * stepMs, WIDE, RADIUS) })
  return fixation as Fixation | null
}

describe('trackFixation', () => {
  it('keeps one fixation while samples stay near its mean', () => {
    const fixation = feed([{ x: 0.5, y: 0.5 }, { x: 0.51, y: 0.46 }, { x: 0.49, y: 0.54 }])
    expect(fixation).toMatchObject({ samples: 3, since: 0 })
    expect(fixation?.x).toBeCloseTo(0.5, 12)
    expect(fixation?.y).toBeCloseTo(0.5, 12)
  })

  it('starts over after a jump', () => {
    expect(feed([{ x: 0.5, y: 0.5 }, { x: 0.5, y: 0.5 }, { x: 0.5, y: 0.7 }])).toMatchObject({ samples: 1, since: 400, y: 0.7 })
  })

  it('measures the radius in screen-height units', () => {
    // 0.05 of 1600 px = 80 px = 0.1 of the 800 px height: too far.
    expect(feed([{ x: 0.5, y: 0.5 }, { x: 0.55, y: 0.5 }])).toMatchObject({ samples: 1 })
    expect(feed([{ x: 0.5, y: 0.5 }, { x: 0.5, y: 0.57 }])).toMatchObject({ samples: 2 })
  })

  it('ends when gaze is lost', () => {
    expect(feed([{ x: 0.5, y: 0.5 }, null])).toBeNull()
    expect(trackFixation(null, { x: NaN, y: 0.5 }, 0, WIDE, RADIUS)).toBeNull()
  })
})

describe('isDwelling', () => {
  it('needs the fixation to last the dwell time', () => {
    const fixation = feed([{ x: 0.5, y: 0.5 }])
    expect(isDwelling(fixation, 599, 600)).toBe(false)
    expect(isDwelling(fixation, 600, 600)).toBe(true)
    expect(isDwelling(null, 10_000, 600)).toBe(false)
  })
})
