import { GazeCrosshair } from '../gaze/GazeCrosshair'
import { useGazeTracking } from '../gaze/useGazeTracking'
import Maze from './Maze'

interface Props {
  onBack: () => void
}

export function MazeScreen({ onBack }: Props) {
  const { videoRef, canvasRef, gazeDirection, gazePoint, statusText } = useGazeTracking()

  return (
    <main className="maze-screen">
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        style={{ display: 'none' }}
      />

      <canvas ref={canvasRef} style={{ display: 'none' }} />

      <Maze gazeDirection={gazeDirection} />

      <GazeCrosshair gaze={gazePoint} />

      <p role="status" aria-live="polite">
        {statusText}
      </p>

      <button
        type="button"
        className="back-button"
        onClick={onBack}
      >
        Back to Home
      </button>
    </main>
  )
}
