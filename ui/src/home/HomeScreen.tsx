import { useState } from 'react'
import type { CalibrationResult } from '../calibration/points'
import './home.css'

interface Props {
  calibration: CalibrationResult | null
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
export function HomeScreen({ calibration, onStartMaze, onStartRace, onStartCalibration, theme, onToggleTheme, }: Props) {
  const [choosingMode, setChoosingMode] = useState(false)

  return (
    <main className="home">
      <h1>Gaze Maze</h1>
      <p className="home-tagline">Steer a ball through a maze using only your eyes.</p>

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
          <button type="button" className="home-card" onClick={() => setChoosingMode(true)}>
            <h2>Maze</h2>
            <p>Play the maze.</p>
          </button>
          <button type="button" className="home-card" onClick={onStartCalibration}>
            <h2>{calibration ? 'Recalibrate' : 'Calibration'}</h2>
            <p>
              {calibration
                ? `${calibration.samples.length} frames captured. Run again to replace them.`
                : 'Teach the model where you are looking. Takes about a minute.'}
            </p>
          </button>
        </div>
      )}

      <p className="home-note">
        Calibration uses your webcam. Frames stay in this browser tab and are never uploaded.
      </p>
      <div className="home-theme">
        <button
          type="button"
          className="theme-toggle"
          onClick={onToggleTheme}
        >
          Switch to {theme === 'light' ? 'dark' : 'light'} mode
        </button>
      </div>
    </main>
  )
}
