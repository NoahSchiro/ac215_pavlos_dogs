import { useState } from 'react'
import type { CalibrationTaskState } from '../calibration/process'
import type { CalibrationSession } from '../calibration/record'
import './home.css'

interface Props {
  calibration: CalibrationSession | null
  calibrationTask: CalibrationTaskState
  onStartValidation: () => void
  onCancelProcessing: () => void
  onClearCalibration: () => void
  onVerifyCamera: () => void
  onStartMaze: () => void
  onStartRace: () => void
  onStartCalibration: () => void
  theme: 'light' | 'dark'
  onToggleTheme: () => void
}

/**
 * Entry screen: choose between the maze and gaze calibration. Picking the maze
 * then asks for a mode: single player or a race against a friend.
 */
export function HomeScreen({
  calibration,
  calibrationTask,
  onStartValidation,
  onCancelProcessing,
  onClearCalibration,
  onVerifyCamera,
  onStartMaze,
  onStartRace,
  onStartCalibration,
  theme,
  onToggleTheme,
}: Props) {
  const [choosingMode, setChoosingMode] = useState(false)

  return (
    <main className="home">
      <h1>Gaze Maze</h1>
      <p className="home-tagline">
        Steer a ball through a maze using only your eyes.
      </p>

      {choosingMode ? (
        <>
          <div className="home-actions" role="group" aria-label="Choose a mode">
            <button type="button" className="home-card" onClick={onStartMaze}>
              <h2>Single player</h2>
              <p>Just you and the maze.</p>
            </button>
            <button type="button" className="home-card" onClick={onStartRace}>
              <h2>Race a friend</h2>
              <p>Two players, one maze. First to the cheese wins.</p>
            </button>
          </div>
          <button type="button" className="home-back" onClick={() => setChoosingMode(false)}>
            ← Back
          </button>
        </>
      ) : (
        <div className="home-actions">
          <button
            type="button"
            className="home-card"
            onClick={() => setChoosingMode(true)}
          >
            <h2>Maze</h2>
            <p>Play the maze.</p>
          </button>

          <button
            type="button"
            className="home-card"
            onClick={onStartCalibration}
          >
            <h2>{calibration ? 'Recalibrate' : 'Calibration'}</h2>
            <p>
              {calibration
                ? `${calibration.record.calibration.validFrameCounts.reduce(
                    (sum, count) => sum + count,
                    0,
                  )} valid frames. Capture again to replace this calibration.`
                : 'Teach the model where you are looking. Takes about a minute.'}
            </p>
          </button>
        </div>
      )}

      <section
        className="calibration-result"
        aria-label="Calibration status"
      >
        {calibrationTask.status === 'restoring' && (
          <p role="status">Checking saved calibration.</p>
        )}

        {calibrationTask.status === 'restored' && (
          <p role="status">
            Saved calibration restored. Verify the camera or recalibrate
            before reuse.
          </p>
        )}

        {calibrationTask.status === 'verifying' && (
          <p role="status">Verifying saved camera.</p>
        )}

        {calibrationTask.status === 'verified' && (
          <p role="status">
            Saved calibration camera verified. Recalibrate if your
            position or posture has changed.
          </p>
        )}

        {calibrationTask.status === 'clearing' && (
          <p role="status">Clearing calibration.</p>
        )}

        {calibrationTask.status === 'cleared' && (
          <p role="status">Calibration cleared.</p>
        )}

        {calibrationTask.status === 'saving' && (
          <>
            <p role="status">Saving calibration.</p>
            <button
              type="button"
              onClick={onCancelProcessing}
            >
              Cancel processing
            </button>
          </>
        )}

        {calibrationTask.status === 'processing' && (
          <>
            <p role="status">
              Processing calibration: {calibrationTask.completed} of{' '}
              {calibrationTask.total} frames.
            </p>
            <progress
              aria-label="Frames processed"
              value={calibrationTask.completed}
              max={calibrationTask.total || 1}
            />
            <button
              type="button"
              onClick={onCancelProcessing}
            >
              Cancel processing
            </button>
          </>
        )}

        {calibrationTask.status === 'success' && (
          <p role="status">
            Calibration ready. {calibrationTask.validFrames} of{' '}
            {calibrationTask.total} frames used. Saved on this device.
          </p>
        )}

        {calibrationTask.status === 'error' && (
          <>
            <p role="alert">
              Calibration failed. {calibrationTask.message}
              {calibration?.cameraVerification === 'verified'
                ? ' Previous calibration kept.'
                : ''}
            </p>
            <button
              type="button"
              onClick={onStartCalibration}
            >
              Retry calibration
            </button>
          </>
        )}

        {calibrationTask.status === 'cancelled' && (
          <p role="status">
            Calibration processing cancelled.
            {calibration ? ' Previous calibration kept.' : ''}
          </p>
        )}

        <div className="calibration-storage-actions">
          {calibration?.cameraVerification === 'pending' && (
            <button
              type="button"
              disabled={calibrationTask.status === 'verifying'}
              onClick={onVerifyCamera}
            >
              Verify saved camera
            </button>
          )}

          <button
            type="button"
            disabled={calibrationTask.status === 'clearing'}
            onClick={onClearCalibration}
          >
            Clear calibration
          </button>

          <button
            type="button"
            disabled={calibration?.cameraVerification !== 'verified'}
            onClick={onStartValidation}
          >
            Validate calibration
          </button>
        </div>
      </section>

      <p className="home-note">
        Calibration uses your webcam. Images are processed in memory
        and are never uploaded or saved. Calibration parameters are
        saved in this browser on this device. Clear calibration
        deletes the saved record.
      </p>

      <div className="home-theme">
        <button
          type="button"
          className="theme-toggle"
          onClick={onToggleTheme}
        >
          <span aria-hidden="true">
            {theme === 'light' ? '🌙' : '☀️'}
          </span>
          <span>
            {theme === 'light' ? 'Dark mode' : 'Light mode'}
          </span>
        </button>
      </div>
    </main>
  )
}
