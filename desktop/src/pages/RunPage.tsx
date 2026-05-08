import { useEffect, useRef, useState } from 'react'
import { useLocation } from 'react-router-dom'
import Card from '../components/Card'
import { Button } from '../components/FormControls'
import { cancelRun, onStream, prepareLocalModel, readBackendStatus, revealPath, runPipeline } from '../lib/bridge'
import type { AppState } from '../App'

const RUN_STATE_KEY = 'transub.runState.v1'

interface Props {
  state: AppState
  updateState: (patch: Partial<AppState>) => void
}

export default function RunPage({ state, updateState }: Props) {
  const location = useLocation()
  const transcribeOnly = (location.state as any)?.transcribeOnly || false
  const initialRunState = loadSavedRunState(state.videoPath)
  const [events, setEvents] = useState<ActivityItem[]>(initialRunState.events)
  const [progress, setProgress] = useState(initialRunState.progress)
  const [status, setStatus] = useState<'idle' | 'preparing' | 'running' | 'done' | 'error'>(initialRunState.status)
  const [activeStage, setActiveStage] = useState(initialRunState.activeStage)
  const [outputPath, setOutputPath] = useState<string | null>(initialRunState.outputPath)
  const [logPath, setLogPath] = useState<string | null>(initialRunState.logPath)
  const eventRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (eventRef.current) eventRef.current.scrollTop = eventRef.current.scrollHeight
  }, [events])

  useEffect(() => {
    saveRunState({
      videoPath: state.videoPath,
      transcribeOnly,
      events,
      progress,
      status,
      activeStage,
      outputPath,
      logPath,
    })
  }, [activeStage, events, logPath, outputPath, progress, state.videoPath, status, transcribeOnly])

  useEffect(() => {
    readBackendStatus().then(current => {
      if (current?.state === 'running') {
        setStatus('running')
        setActiveStage(current.stage || 'running')
        setProgress(Number(current.progress || 0))
        setLogPath(current.log_path || null)
        updateState({ isRunning: true })
      } else if (current?.state === 'error') {
        setStatus('error')
        setActiveStage(current.stage || initialRunState.activeStage || 'error')
        setProgress(Number(current.progress || 0))
        setLogPath(current.log_path || null)
        setEvents(prev => prev.length > 0 ? prev : [{
          kind: 'error',
          title: 'Previous run failed',
          detail: current.message || 'Fix configuration and run again.',
        }])
      } else if (current?.state === 'done') {
        setStatus('done')
        setActiveStage('done')
        setProgress(100)
        setLogPath(current.log_path || null)
      }
    })
  }, [updateState])

  useEffect(() => {
    const unsub = onStream((data: any) => {
      if (data.type === 'progress') {
        setEvents(prev => appendActivity(prev, {
          kind: 'progress',
          title: stageLabel(data.stage),
          detail: data.message,
        }))
        setActiveStage(data.stage || 'running')
        if (data.log_path) setLogPath(data.log_path)
        if (typeof data.percent === 'number') {
          setProgress(previous => Math.max(previous, Math.min(99, data.percent)))
        }
        if (data.done && data.total) {
          setProgress(previous => Math.max(previous, Math.round((data.done / data.total) * 86)))
        }
      } else if (data.type === 'done') {
        if (data.log_path) setLogPath(data.log_path)
        if (data.success) {
          setOutputPath(data.output_path || null)
          setEvents(prev => appendActivity(prev, {
            kind: 'done',
            title: 'Finished',
            detail: data.output_path ? `Saved to ${data.output_path}` : 'Subtitle file exported.',
          }))
          setProgress(100)
          setStatus('done')
          setActiveStage('done')
        } else {
          setEvents(prev => appendActivity(prev, {
            kind: 'error',
            title: 'Run failed',
            detail: data.error || 'The backend stopped before finishing.',
          }))
          setStatus('error')
        }
        updateState({ isRunning: false })
      } else if (data.type === 'error') {
        setEvents(prev => appendActivity(prev, {
          kind: 'error',
          title: 'Connection issue',
          detail: data.error || 'Connection failed',
        }))
        setStatus('error')
        updateState({ isRunning: false })
      }
    })
    return unsub
  }, [updateState])

  const handlePrepare = async () => {
    setEvents([
      {
        kind: 'progress',
        title: 'Preparing model',
        detail: 'First local runs may download model files. This is a one-time setup step.',
      },
    ])
    setProgress(0)
    setStatus('preparing')
    setActiveStage('prepare')
    updateState({ isRunning: true })

    const result = await prepareLocalModel(state.configPath || undefined)
    if (result.status === 'ok') {
      setEvents(prev => appendActivity(prev, { kind: 'done', title: 'Model ready', detail: result.message }))
      setStatus('idle')
      setActiveStage('ready')
    } else {
      setEvents(prev => appendActivity(prev, { kind: 'error', title: 'Prepare failed', detail: result.detail || 'Failed' }))
      setStatus('error')
      setActiveStage('error')
    }
    updateState({ isRunning: false })
  }

  const handleRun = async () => {
    if (!state.videoPath) return
    setEvents([
      { kind: 'progress', title: 'Queued', detail: `Input: ${state.videoPath}` },
      { kind: 'progress', title: 'Starting backend', detail: 'The local Transub service is preparing this job.' },
    ])
    setProgress(0)
    setStatus('running')
    setActiveStage('queued')
    setOutputPath(null)
    setLogPath(null)
    updateState({ isRunning: true })

    const result = await runPipeline(state.videoPath, {
      transcribeOnly,
      configPath: state.configPath || undefined,
    })

    if (result.status !== 'started') {
      setEvents(prev => appendActivity(prev, { kind: 'error', title: 'Could not start', detail: result.detail || 'Failed to start' }))
      setStatus('error')
      setActiveStage('error')
      updateState({ isRunning: false })
    } else {
      setLogPath(result.log_path || null)
      setEvents(prev => appendActivity(prev, { kind: 'progress', title: 'Run started', detail: 'Progress will update here. Detailed logs are saved separately.' }))
    }
  }

  const handleCancel = async () => {
    await cancelRun()
    setStatus('idle')
    setActiveStage('ready')
    setEvents(prev => appendActivity(prev, { kind: 'progress', title: 'Cancelling', detail: 'The backend will stop at the next safe checkpoint.' }))
    updateState({ isRunning: false })
  }

  const statusLabel = {
    idle: 'Ready',
    preparing: 'Preparing model',
    running: 'Running',
    done: 'Completed',
    error: 'Failed',
  }[status]

  const fileName = state.videoPath ? state.videoPath.split(/[/\\]/).pop() : 'No video selected'
  const stages = buildStages(transcribeOnly, activeStage, status)

  return (
    <div className="page">
      <section className="section-header">
        <div>
          <h1 className="page-title">{transcribeOnly ? 'Transcription Run' : 'Pipeline Run'}</h1>
          <p className="page-kicker">
            Real-time pipeline progress: extraction, transcription, translation, export.
          </p>
        </div>
        <div className={`status-pill ${status === 'done' ? 'ready' : status === 'error' ? 'error' : status === 'running' ? 'warning' : ''}`}>
          {statusLabel}
        </div>
      </section>

      <div className="dashboard-grid run-dashboard">
        <Card icon={<StepIcon label="1" />} title="Input" description={fileName} status={state.videoPath ? 'ready' : 'pending'} />
        <Card icon={<StepIcon label="2" />} title="Local model" description="faster-whisper" status={status === 'preparing' ? 'warning' : status === 'error' && activeStage === 'prepare' ? 'error' : 'pending'} />
        <Card icon={<StepIcon label="3" />} title="Output" description={transcribeOnly ? 'Source-language subtitles' : 'Translated subtitles'} status={status === 'done' ? 'ready' : 'pending'} />
        <Card icon={<StepIcon label="4" />} title="Debug log" description={logPath ? 'Saved locally' : 'Created when a run starts'} status={logPath ? 'ready' : 'pending'} />
      </div>

      <section className="run-panel">
        <div className="section-header">
          <div>
            <h2 className="section-title">Pipeline Console</h2>
            <p className="page-kicker">{state.videoPath || 'Select a video before starting a run.'}</p>
          </div>
          <div className="button-row">
            {status === 'running' || status === 'preparing' ? (
              <Button variant="danger" onClick={handleCancel}>Cancel</Button>
            ) : (
              <>
                <Button onClick={handlePrepare} disabled={state.isRunning}>
                  Prepare local model
                </Button>
                <Button variant="primary" disabled={!state.videoPath || state.isRunning} onClick={handleRun}>
                  {status === 'done' ? 'Run again' : transcribeOnly ? 'Start transcription' : 'Start pipeline'}
                </Button>
              </>
            )}
          </div>
        </div>

        <div className="run-progress-card">
          <div className="progress-track">
            <div className="progress-fill" style={{ width: `${progress}%` }} />
          </div>
          <div className="run-progress-meta">
            <span>{statusLabel}</span>
            <strong>{progress}%</strong>
          </div>
        </div>

        <div className="run-workspace">
          <div className="stage-board">
            {stages.map(stage => (
              <div key={stage.id} className={`stage-row ${stage.state}`}>
                <span className="stage-marker">{stage.state === 'done' ? '✓' : stage.state === 'active' ? '•' : stage.state === 'error' ? '!' : ''}</span>
                <div>
                  <strong>{stage.title}</strong>
                  <span>{stage.detail}</span>
                </div>
              </div>
            ))}
          </div>

          <aside className="run-inspector">
            <h3>Run files</h3>
            <FileAction
              label="Output"
              value={outputPath || 'Not exported yet'}
              disabled={!outputPath}
              onReveal={() => outputPath && revealPath(outputPath)}
            />
            <FileAction
              label="Debug log"
              value={logPath || 'Created after the job starts'}
              disabled={!logPath}
              onReveal={() => logPath && revealPath(logPath)}
            />
            <div className="native-note">
              The interface shows the human-readable run state. Detailed backend messages are still written to the debug log for troubleshooting.
            </div>
          </aside>
        </div>

        <div ref={eventRef} className="activity-feed">
          {events.length === 0 ? (
            <div className="empty-activity">Run activity will appear here.</div>
          ) : events.map((item, index) => (
            <div key={`${item.title}-${index}`} className={`activity-item ${item.kind}`}>
              <strong>{item.title}</strong>
              <span>{item.detail}</span>
            </div>
          ))}
        </div>
      </section>
    </div>
  )
}

interface SavedRunState {
  videoPath: string | null
  transcribeOnly: boolean
  events: ActivityItem[]
  progress: number
  status: 'idle' | 'preparing' | 'running' | 'done' | 'error'
  activeStage: string
  outputPath: string | null
  logPath: string | null
}

function loadSavedRunState(videoPath: string | null): SavedRunState {
  const fallback: SavedRunState = {
    videoPath,
    transcribeOnly: false,
    events: [],
    progress: 0,
    status: 'idle',
    activeStage: 'ready',
    outputPath: null,
    logPath: null,
  }
  try {
    const saved = window.localStorage.getItem(RUN_STATE_KEY)
    if (!saved) return fallback
    const parsed = JSON.parse(saved)
    if (parsed.videoPath && videoPath && parsed.videoPath !== videoPath) return fallback
    return { ...fallback, ...parsed, status: parsed.status === 'running' ? 'idle' : parsed.status }
  } catch {
    return fallback
  }
}

function saveRunState(state: SavedRunState) {
  try {
    window.localStorage.setItem(RUN_STATE_KEY, JSON.stringify(state))
  } catch {
    // Ignore persistence failures; the live run state still works.
  }
}

function StepIcon({ label }: { label: string }) {
  return <strong>{label}</strong>
}

interface ActivityItem {
  kind: 'progress' | 'done' | 'error'
  title: string
  detail: string
}

function appendActivity(items: ActivityItem[], item: ActivityItem) {
  const previous = items[items.length - 1]
  if (previous && previous.title === item.title && previous.detail === item.detail && previous.kind === item.kind) {
    return items
  }
  return [...items.slice(-10), item]
}

function stageLabel(stage: string) {
  const labels: Record<string, string> = {
    queued: 'Queued',
    extracting: 'Extract audio',
    transcribing: 'Transcribe',
    optimizing: 'Optimize',
    translating: 'Translate',
    polishing: 'Polish',
  }
  return labels[stage] || 'Working'
}

function buildStages(transcribeOnly: boolean, activeStage: string, status: string) {
  const ids = transcribeOnly
    ? ['queued', 'extracting', 'transcribing', 'done']
    : ['queued', 'extracting', 'transcribing', 'translating', 'polishing', 'done']
  const labels: Record<string, { title: string; detail: string }> = {
    queued: { title: 'Prepare job', detail: 'Validate input and create a local run record.' },
    extracting: { title: 'Extract audio', detail: 'Generate a clean audio file for ASR.' },
    transcribing: { title: 'Transcribe speech', detail: 'Use faster-whisper with timestamps.' },
    translating: { title: 'Translate subtitles', detail: 'Send subtitle batches to the selected provider.' },
    polishing: { title: 'Polish and export', detail: 'Refine line breaks and write the subtitle file.' },
    done: { title: 'Ready to use', detail: 'Open the exported subtitle next to your video.' },
  }
  const activeIndex = ids.indexOf(activeStage)
  const doneIndex = status === 'done' ? ids.length - 1 : activeIndex
  return ids.map((id, index) => ({
    id,
    ...labels[id],
    state: status === 'error' && id === activeStage
      ? 'error'
      : status === 'done' || (doneIndex >= 0 && index < doneIndex)
        ? 'done'
        : id === activeStage || (status === 'running' && activeIndex === -1 && index === 0)
          ? 'active'
          : 'pending',
  }))
}

function FileAction({
  label,
  value,
  disabled,
  onReveal,
}: {
  label: string
  value: string
  disabled: boolean
  onReveal: () => void
}) {
  return (
    <div className="file-action">
      <div>
        <span>{label}</span>
        <strong>{value}</strong>
      </div>
      <Button size="sm" disabled={disabled} onClick={onReveal}>Show</Button>
    </div>
  )
}
