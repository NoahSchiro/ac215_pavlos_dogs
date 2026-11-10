import { useCallback, useEffect, useState } from 'react'
import { Calibration } from './calibration/Calibration'
import { exitFullscreen } from './calibration/fullscreen'
import type { CalibrationResult } from './calibration/points'
import { HomeScreen } from './home/HomeScreen'
import { MazeScreen } from './maze/MazeScreen'
import { RaceScreen } from './race/RaceScreen'

type View = 'home' | 'maze' | 'race' | 'calibration'

function App() {
  // Theme controls the appearance.
  const [theme, setTheme] = useState<'light' | 'dark'>('light')

  useEffect(() => {
    document.documentElement.dataset.theme = theme
  }, [theme])

  // View controls which screen is displayed.
  const [view, setView] = useState<View>('home')
  const [calibration, setCalibration] =
    useState<CalibrationResult | null>(null)

  const goHome = useCallback(() => setView('home'), [])

  // Calibration handles full screen after camera permission.
  const startCalibration = useCallback(
    () => setView('calibration'),
    [],
  )

  const finishCalibration = useCallback((result: CalibrationResult) => {
    setCalibration(result)
    void exitFullscreen()
    setView('home')
  }, [])

  const cancelCalibration = useCallback(() => {
    void exitFullscreen()
    setView('home')
  }, [])

  // Keep calibration's screen layout unchanged.
  if (view === 'calibration') {
    return (
      <Calibration
        onComplete={finishCalibration}
        onCancel={cancelCalibration}
      />
    )
  }

  return (
    <>
      {view === 'maze' ? (
        <MazeScreen onBack={goHome} />
      ) : view === 'race' ? (
        <RaceScreen onBack={goHome} />
      ) : (
        <HomeScreen
          calibration={calibration}
          onStartMaze={() => setView('maze')}
          onStartRace={() => setView('race')}
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