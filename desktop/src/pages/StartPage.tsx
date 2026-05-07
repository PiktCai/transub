import { useNavigate } from 'react-router-dom'
import { Button } from '../components/FormControls'
import { selectVideoFile } from '../lib/bridge'
import type { AppState } from '../App'

interface Props {
  state: AppState
  updateState: (patch: Partial<AppState>) => void
}

export default function StartPage({ state, updateState }: Props) {
  const navigate = useNavigate()
  const fileName = state.videoPath ? state.videoPath.split(/[/\\]/).pop() : null

  const handleSelect = async () => {
    const path = await selectVideoFile()
    if (path) updateState({ videoPath: path })
  }

  return (
    <div className="native-page">
      <section className="native-toolbar">
        <div>
          <h1>New subtitle job</h1>
          <p>{fileName || 'No source video selected'}</p>
        </div>
        <div className="button-row">
          <Button onClick={() => navigate('/setup')}>Setup</Button>
          <Button onClick={() => navigate('/providers')}>Keys</Button>
          <Button variant="primary" onClick={handleSelect}>Choose source</Button>
        </div>
      </section>

      <section className="job-layout">
        <div className="native-panel source-panel">
          <div className="panel-titlebar">
            <h2>Source</h2>
            <span className={`status-pill ${state.videoPath ? 'ready' : 'warning'}`}>
              {state.videoPath ? 'Selected' : 'Required'}
            </span>
          </div>

          <button className="source-dropzone" onClick={handleSelect}>
            <div className="source-icon"><MediaIcon /></div>
            <div>
              <strong>{fileName || 'Choose a video file'}</strong>
              <span>{state.videoPath || 'MP4, MOV, MKV, WebM, AVI'}</span>
            </div>
          </button>

          <div className="property-list">
            <PropertyRow label="Transcription" value="Fast local transcription" />
            <PropertyRow label="Translation" value="OpenAI-compatible provider" />
            <PropertyRow label="Output" value="SRT, next to source" />
          </div>
        </div>

        <div className="native-panel">
          <div className="panel-titlebar">
            <h2>Readiness</h2>
            <span className="status-pill">4 checks</span>
          </div>
          <div className="native-list">
            <CheckRow
              done={!!state.videoPath}
              title="Input file"
              detail={fileName || 'Choose a source video before running'}
              actionLabel={state.videoPath ? 'Change' : 'Choose'}
              onAction={handleSelect}
            />
            <CheckRow done title="Engine" detail="Default local ASR is selected" actionLabel="Review" onAction={() => navigate('/setup')} />
            <CheckRow done={false} title="Provider key" detail="Save a key for the translation provider" actionLabel="Open" onAction={() => navigate('/providers')} />
            <CheckRow done={false} title="Dry run" detail="Run transcription-only before spending API tokens" actionLabel="Run" onAction={() => navigate('/run', { state: { transcribeOnly: true } })} />
          </div>
        </div>

        <div className="native-panel inspector-panel">
          <div className="panel-titlebar">
            <h2>Job actions</h2>
          </div>
          <div className="action-stack">
            <Button variant="primary" disabled={!state.videoPath} onClick={() => navigate('/run')}>Run pipeline</Button>
            <Button disabled={!state.videoPath} onClick={() => navigate('/run', { state: { transcribeOnly: true } })}>Transcribe only</Button>
            <Button onClick={() => navigate('/subtitles')}>Open subtitle preview</Button>
          </div>
          <div className="native-note">
            Transub runs the same CLI pipeline underneath this desktop shell. The GUI keeps the working set visible instead of hiding it behind a wizard.
          </div>
        </div>
      </section>
    </div>
  )
}

function PropertyRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="property-row">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  )
}

function CheckRow({
  done,
  title,
  detail,
  actionLabel,
  onAction,
}: {
  done: boolean
  title: string
  detail: string
  actionLabel: string
  onAction: () => void
}) {
  return (
    <div className="native-list-row">
      <span className={`checkmark ${done ? 'is-done' : ''}`}>{done ? '✓' : '·'}</span>
      <div>
        <strong>{title}</strong>
        <span>{detail}</span>
      </div>
      <button onClick={onAction}>{actionLabel}</button>
    </div>
  )
}

function MediaIcon() {
  return (
    <svg width="26" height="26" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path d="M5 7h14v10H5zM9 7l2 4M14 7l2 4M5 11h14" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  )
}
