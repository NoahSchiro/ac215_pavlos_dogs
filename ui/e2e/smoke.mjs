/**
 * Smoke test of what users actually run: the production UI, a real browser, a
 * webcam showing a face. Passes when the maze reaches "Tracking gaze", keeps
 * tracking, and the model came back 200.
 *
 * This is the gap the unit suites leave: they run against the Vite dev server
 * (and CI skips the vision suite entirely), which is how the production-only
 * OpenCV loading bug shipped. Run it through deploy/smoke-test.sh.
 *
 *   node smoke.mjs <url> <face.mjpeg>
 */
import { chromium } from 'playwright'

const [url, face] = process.argv.slice(2)
const TRACK_TIMEOUT_S = 60
const HOLD_S = 10

const browser = await chromium.launch({
  args: [
    '--use-fake-ui-for-media-stream',
    '--use-fake-device-for-media-stream',
    // Chrome plays an MJPEG file as the camera; one JPEG is a one-frame MJPEG.
    `--use-file-for-fake-video-capture=${face}`,
  ],
})
const page = await (await browser.newContext({ permissions: ['camera'] })).newPage()
const models = []
page.on('requestfinished', async (r) => {
  if (r.url().endsWith('.onnx')) models.push(`${(await r.response())?.status()} ${r.url()}`)
})
page.on('requestfailed', (r) => {
  if (r.url().endsWith('.onnx')) models.push(`FAILED ${r.failure()?.errorText} ${r.url()}`)
})

const timeline = []
const status = async () => (await page.locator('[role=status]').first().textContent().catch(() => '')) ?? ''
const note = (s) => { if (timeline.at(-1)?.[1] !== s) timeline.push([((Date.now() - t0) / 1000).toFixed(1), s]) }

const t0 = Date.now()
await page.goto(url, { waitUntil: 'load', timeout: 120_000 })
await page.locator('button.home-card').first().click()

let tracking = false
while (!tracking && Date.now() - t0 < TRACK_TIMEOUT_S * 1000) {
  const s = await status()
  note(s)
  tracking = s.startsWith('Tracking gaze')
  if (!tracking) await page.waitForTimeout(500)
}
let held = tracking
for (let i = 0; held && i < HOLD_S * 2; i++) {
  await page.waitForTimeout(500)
  const s = await status()
  note(s)
  held = s.startsWith('Tracking gaze')
}
await browser.close()

for (const [t, s] of timeline) console.log(`  t=${t.padStart(5)}s  ${s}`)
console.log(`  model: ${models.join(' | ') || 'not requested'}`)
const modelOk = models.some((m) => m.startsWith('200 '))
const pass = tracking && held && modelOk
console.log(pass ? 'SMOKE TEST PASSED' : `SMOKE TEST FAILED (${!modelOk ? 'model did not load' : !tracking ? 'never started tracking' : 'tracking did not hold'})`)
process.exit(pass ? 0 : 1)
