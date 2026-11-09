import { useEffect, useRef, useState } from 'react'
import { Calibration } from './Calibration'
import { exitFullscreen } from './fullscreen'
import type { CalibrationResult } from './points'
import type { CalibrationSession } from './record'
import { validateCalibrationCapture, type ValidationSummary } from './validation'
import { captureViewport } from './viewport'

type State = { status: 'capture' } | { status: 'processing'; completed: number }
  | { status: 'done'; summary: ValidationSummary } | { status: 'error'; message: string }

/** A temporary check: never saves samples, refits, or replaces calibration. */
export function AccuracyCheck({ session, onBack }: { session: CalibrationSession; onBack: () => void }) {
  const [state, setState] = useState<State>({ status: 'capture' })
  const controller = useRef<AbortController | null>(null)
  useEffect(() => () => controller.current?.abort(), [])

  function leave() {
    controller.current?.abort()
    void exitFullscreen()
    onBack()
  }

  async function score(capture: CalibrationResult) {
    controller.current?.abort()
    const task = new AbortController()
    controller.current = task
    void exitFullscreen()
    setState({ status: 'processing', completed: 0 })
    try {
      const summary = await validateCalibrationCapture(capture, session, {
        signal: task.signal, onProgress: (completed) => setState({ status: 'processing', completed }),
      })
      if (!task.signal.aborted) setState({ status: 'done', summary })
    } catch (error) {
      if (!task.signal.aborted) setState({ status: 'error', message: error instanceof Error ? error.message : String(error) })
    }
  }

  if (state.status === 'capture') {
    return <Calibration mode="validation" expectedViewport={captureViewport(session.record.calibration.capture)}
      onComplete={(capture) => { void score(capture) }} onCancel={leave} />
  }
  return (
    <main className="home">
      <h1>Accuracy check</h1>
      <section className="calibration-result" aria-label="Calibration accuracy">
        {state.status === 'processing' && <p role="status">Checking accuracy: {state.completed} of 12 frames.</p>}
        {state.status === 'error' && <p role="alert">{state.message}</p>}
        {state.status === 'done' && <>
          <p>Raw mean error: {state.summary.rawMeanError.toFixed(6)}</p>
          <p>Corrected mean error: {state.summary.correctedMeanError.toFixed(6)}</p>
          <p>{state.summary.validFrameCount} of 12 frames used. Lower error is better.</p>
          <p>{state.summary.correctedMeanError < state.summary.rawMeanError
            ? 'Calibration improved this check.' : 'Calibration did not improve this check. Try recalibrating.'}</p>
        </>}
        <p>Mean error in normalized screen coordinates, without clipping. This check does not change your calibration.</p>
        <div className="calibration-storage-actions">
          {state.status !== 'processing' && <button type="button" onClick={() => setState({ status: 'capture' })}>Try again</button>}
          <button type="button" onClick={leave}>Back to Home</button>
        </div>
      </section>
    </main>
  )
}
