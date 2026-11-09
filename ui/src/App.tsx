import { useCallback, useEffect, useRef, useState } from 'react'
import { Calibration } from './calibration/Calibration'
import { AccuracyCheck } from './calibration/AccuracyCheck'
import { exitFullscreen } from './calibration/fullscreen'
import type { CalibrationResult } from './calibration/points'
import { processCalibration, type CalibrationTaskState } from './calibration/process'
import { createCalibrationRecord, type CalibrationRecord, type CalibrationSession } from './calibration/record'
import { clearCalibration, loadCalibration, saveCalibration } from './calibration/storage'
import { verifyCaptureCamera } from './calibration/cameraMetadata'
import { HomeScreen } from './home/HomeScreen'
import { MazeScreen } from './maze/MazeScreen'

type View = 'home' | 'maze' | 'calibration' | 'validation'

function App() {
  // Theme controls the appearance.
  const [theme, setTheme] = useState<'light' | 'dark'>('light')

  useEffect(() => {
    document.documentElement.dataset.theme = theme
  }, [theme])

  // View controls which screen is displayed.
  const [view, setView] = useState<View>('home')
  const [calibration, setCalibration] =
    useState<CalibrationSession | null>(null)
  const [calibrationTask, setCalibrationTask] = useState<CalibrationTaskState>({ status: 'restoring' })
  const taskIdRef = useRef(0)
  const controllerRef = useRef<AbortController | null>(null)

  const invalidateTask = useCallback(() => {
    taskIdRef.current++
    controllerRef.current?.abort()
    controllerRef.current = null
    return taskIdRef.current
  }, [])

  useEffect(() => {
    const taskId = taskIdRef.current
    const controller = new AbortController()
    void loadCalibration(controller.signal).then((result) => {
      if (controller.signal.aborted || taskId !== taskIdRef.current) return
      if (result.status === 'restored') {
        setCalibration({ record: result.record, cameraVerification: 'pending' })
        setCalibrationTask({ status: 'restored' })
      } else if (result.status === 'invalid') {
        setCalibrationTask({ status: 'error', message: result.reason })
      } else setCalibrationTask({ status: 'idle' })
    }).catch((error) => {
      if (controller.signal.aborted || taskId !== taskIdRef.current) return
      setCalibrationTask({ status: 'error', message: `Could not restore calibration: ${error instanceof Error ? error.message : String(error)}` })
    })
    return () => { controller.abort(); invalidateTask() }
  }, [invalidateTask])

  const goHome = useCallback(() => setView('home'), [])

  // Calibration handles full screen after camera permission.
  const startCalibration = useCallback(() => {
    invalidateTask()
    setCalibrationTask({ status: 'idle' })
    setView('calibration')
  }, [invalidateTask])

  const finishCalibration = useCallback(async (capture: CalibrationResult, captureTaskId: number) => {
    if (captureTaskId !== taskIdRef.current) return
    const taskId = invalidateTask()
    const controller = new AbortController()
    controllerRef.current = controller
    void exitFullscreen()
    setView('home')
    setCalibrationTask({ status: 'processing', completed: 0, total: capture.samples.length })
    try {
      const result = await processCalibration(capture, {
        signal: controller.signal,
        onProgress: (progress) => {
          if (taskId === taskIdRef.current && !controller.signal.aborted) {
            setCalibrationTask({ status: 'processing', ...progress })
          }
        },
      })
      if (taskId !== taskIdRef.current || controller.signal.aborted) return
      if (result.ok) {
        let record: CalibrationRecord
        try {
          record = createCalibrationRecord(result.calibration)
          setCalibrationTask({ status: 'saving' })
          await saveCalibration(record, controller.signal)
        } catch (error) {
          if (taskId !== taskIdRef.current || controller.signal.aborted) return
          setCalibrationTask({ status: 'error', message: `New calibration was not saved: ${error instanceof Error ? error.message : String(error)}` })
          return
        }
        if (taskId !== taskIdRef.current || controller.signal.aborted) return
        setCalibration({ record, cameraVerification: 'verified' })
        setCalibrationTask({
          status: 'success', total: capture.samples.length,
          validFrames: result.calibration.validFrameCounts.reduce((sum, count) => sum + count, 0),
        })
      } else {
        setCalibrationTask({ status: 'error', message: result.reason })
      }
    } catch (error) {
      if (taskId !== taskIdRef.current || controller.signal.aborted) return
      setCalibrationTask({ status: 'error', message: error instanceof Error ? error.message : String(error) })
    } finally {
      if (taskId === taskIdRef.current) controllerRef.current = null
    }
  }, [invalidateTask])

  const cancelCalibration = useCallback((captureTaskId: number) => {
    if (captureTaskId !== taskIdRef.current) return
    invalidateTask()
    void exitFullscreen()
    setView('home')
  }, [invalidateTask])

  const cancelProcessing = useCallback(() => {
    invalidateTask()
    setCalibrationTask({ status: 'cancelled' })
  }, [invalidateTask])

  const startMaze = useCallback(() => {
    invalidateTask()
    setCalibrationTask({ status: 'idle' })
    setView('maze')
  }, [invalidateTask])

  const clearSavedCalibration = useCallback(async () => {
    const taskId = invalidateTask()
    const controller = new AbortController()
    controllerRef.current = controller
    setCalibrationTask({ status: 'clearing' })
    try {
      await clearCalibration(controller.signal)
      if (taskId !== taskIdRef.current || controller.signal.aborted) return
      setCalibration(null)
      setCalibrationTask({ status: 'cleared' })
    } catch (error) {
      if (taskId !== taskIdRef.current || controller.signal.aborted) return
      setCalibrationTask({ status: 'error', message: `Could not clear calibration: ${error instanceof Error ? error.message : String(error)}` })
    } finally {
      if (taskId === taskIdRef.current) controllerRef.current = null
    }
  }, [invalidateTask])

  const verifySavedCamera = useCallback(async () => {
    const camera = calibration?.record.calibration.capture.camera
    if (!camera) return
    const taskId = invalidateTask()
    const controller = new AbortController()
    controllerRef.current = controller
    setCalibrationTask({ status: 'verifying' })
    try {
      await verifyCaptureCamera(camera, controller.signal)
      if (taskId !== taskIdRef.current || controller.signal.aborted) return
      setCalibration({ record: calibration.record, cameraVerification: 'verified' })
      setCalibrationTask({ status: 'verified' })
    } catch (error) {
      if (taskId !== taskIdRef.current || controller.signal.aborted) return
      setCalibrationTask({ status: 'error', message: `Saved camera could not be verified: ${error instanceof Error ? error.message : String(error)} Recalibrate before reuse.` })
    } finally {
      if (taskId === taskIdRef.current) controllerRef.current = null
    }
  }, [calibration, invalidateTask])

  if (view === 'validation' && calibration) return <AccuracyCheck session={calibration} onBack={goHome} />

  // Keep calibration's screen layout unchanged.
  if (view === 'calibration') {
    const captureTaskId = taskIdRef.current
    return (
      <Calibration
        onComplete={(result) => { void finishCalibration(result, captureTaskId) }}
        onCancel={() => cancelCalibration(captureTaskId)}
      />
    )
  }

  return (
    <>
      {view === 'maze' ? (
        <MazeScreen onBack={goHome} calibration={calibration} />
      ) : (
        <HomeScreen
          calibration={calibration}
          calibrationTask={calibrationTask}
          onStartValidation={() => {
            if (calibration?.cameraVerification !== 'verified') return
            invalidateTask()
            setCalibrationTask({ status: 'idle' })
            setView('validation')
          }}
          onCancelProcessing={cancelProcessing}
          onClearCalibration={() => { void clearSavedCalibration() }}
          onVerifyCamera={() => { void verifySavedCamera() }}
          onStartMaze={startMaze}
          onStartCalibration={startCalibration}
          theme={theme}
          onToggleTheme={() =>
            setTheme((current) => (current === 'light' ? 'dark' : 'light'))
          }
        />
      )}
    </>
  )
}

export default App
