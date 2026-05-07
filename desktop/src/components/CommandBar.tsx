import { selectVideoFile } from '../lib/bridge'
import { Button } from './FormControls'

interface CommandBarProps {
  videoPath: string | null
  isRunning: boolean
  onVideoSelect: (path: string) => void
  onRun: () => void
  onTranscribeOnly: () => void
}

export default function CommandBar({
  videoPath,
  isRunning,
  onVideoSelect,
  onRun,
  onTranscribeOnly,
}: CommandBarProps) {
  const handleSelect = async () => {
    const path = await selectVideoFile()
    if (path) onVideoSelect(path)
  }

  const fileName = videoPath ? videoPath.split(/[/\\]/).pop() : 'No video selected'

  return (
    <header className="command-bar">
      <div className="brand-lockup">
        <div className="brand-mark" aria-hidden="true">
          <svg width="28" height="28" viewBox="0 0 28 28" fill="none">
            <rect x="5" y="7" width="18" height="14" rx="2" stroke="currentColor" strokeWidth="2" />
            <path d="M9 12h10M9 16h6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
          </svg>
        </div>
        <div>
          <h1 className="brand-title">Transub</h1>
          <div className="brand-subtitle">Transcribe, translate, export.</div>
        </div>
      </div>

      <div className="video-summary">
        <div className="video-name">{fileName}</div>
        <div className="video-path">{videoPath || 'Choose a video, pick an engine, then run.'}</div>
      </div>

      <div className="command-actions">
        <Button onClick={handleSelect}>
          <FolderIcon /> Select Video
        </Button>
        <Button variant="primary" onClick={onRun} disabled={!videoPath || isRunning}>
          <PlayIcon /> Run Pipeline
        </Button>
        <Button variant="accent" onClick={onTranscribeOnly} disabled={!videoPath || isRunning}>
          Transcribe Only
        </Button>
      </div>
    </header>
  )
}

function FolderIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path d="M4 7.5h6l1.8 2H20v8.5H4z" stroke="currentColor" strokeWidth="2" strokeLinejoin="round" />
    </svg>
  )
}

function PlayIcon() {
  return (
    <svg width="17" height="17" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path d="m8 5 11 7-11 7V5Z" fill="currentColor" />
    </svg>
  )
}
