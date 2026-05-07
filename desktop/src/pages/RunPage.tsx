import { useEffect, useRef, useState } from 'react'
import { useLocation } from 'react-router-dom'
import Card from '../components/Card'
import { Button } from '../components/FormControls'
import { cancelRun, onStream, prepareLocalModel, runPipeline } from '../lib/bridge'
import type { AppState } from '../App'

interface Props {
  state: AppState
  updateState: (patch: Partial<AppState>) => void
}

export default function RunPage({ state, updateState }: Props) {
  const location = useLocation()
  const transcribeOnly = (location.state as any)?.transcribeOnly || false
  const [logs, setLogs] = useState<string[]>([])
  const [progress, setProgress] = useState(0)
  const [status, setStatus] = useState<'idle' | 'preparing' | 'running' | 'done' | 'error'>('idle')
  const logRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight
  }, [logs])

  useEffect(() => {
    const unsub = onStream((data: any) => {
      if (data.type === 'stdout') {
        setLogs(previous => [...previous, data.data])
        const match = data.data.match(/(\d+)%/)
        if (match) setProgress(Number(match[1]))
      } else if (data.type === 'stderr') {
        setLogs(previous => [...previous, formatStderr(data.data)])
      } else if (data.type === 'done') {
        setStatus(data.code === 0 ? 'done' : 'error')
        updateState({ isRunning: false })
        setProgress(data.code === 0 ? 100 : progress)
      } else if (data.type === 'error') {
        setLogs(previous => [...previous, `[error] ${data.error || 'Process failed to start.'}\n`])
        setStatus('error')
        updateState({ isRunning: false })
      }
    })
    return unsub
  }, [progress, updateState])

  const handlePrepare = async () => {
    setLogs([
      'Preparing local transcription model.\n',
      'If this is your first local run, Transub may download model files now. Keep this window open; later runs will use the cache.\n\n',
    ])
    setProgress(0)
    setStatus('preparing')
    updateState({ isRunning: true })

    const result = await prepareLocalModel(state.configPath || undefined)
    if (!result.success) {
      setStatus('error')
      updateState({ isRunning: false })
    }
  }

  const handleRun = async () => {
    if (!state.videoPath) return
    setLogs([
      `Input: ${state.videoPath}\n`,
      'Starting pipeline. Paths with spaces are passed safely to the backend.\n',
      'First local model use can take a few minutes while model files are cached.\n\n',
    ])
    setProgress(0)
    setStatus('running')
    updateState({ isRunning: true })

    const result = await runPipeline(state.videoPath, {
      transcribeOnly,
      configPath: state.configPath || undefined,
    })

    if (!result.success && status === 'running') {
      setStatus('error')
      updateState({ isRunning: false })
    }
  }

  const handleCancel = async () => {
    await cancelRun()
    setStatus('idle')
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

  return (
    <div className="page">
      <section className="section-header">
        <div>
          <h1 className="page-title">{transcribeOnly ? 'Transcription Run' : 'Pipeline Run'}</h1>
          <p className="page-kicker">
            Watch the real CLI process here: extraction, transcription, translation, export, and errors.
          </p>
        </div>
        <div className={`status-pill ${status === 'done' ? 'ready' : status === 'error' ? 'error' : status === 'running' ? 'warning' : ''}`}>
          {statusLabel}
        </div>
      </section>

      <div className="dashboard-grid">
        <Card icon={<StepIcon label="1" />} title="Input" description={fileName} status={state.videoPath ? 'ready' : 'pending'} />
        <Card icon={<StepIcon label="2" />} title="Extract audio" description="ffmpeg working file" status={status === 'running' ? 'warning' : 'pending'} />
        <Card icon={<StepIcon label="3" />} title="Transcribe" description="Prepare local model before first run" status={status === 'preparing' ? 'warning' : status === 'done' ? 'ready' : 'pending'} />
        <Card icon={<StepIcon label="4" />} title={transcribeOnly ? 'Skip translate' : 'Translate'} description={transcribeOnly ? 'Dry run mode' : 'Provider API'} status={transcribeOnly ? 'warning' : 'pending'} />
      </div>

      <section className="console-panel">
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

        <div style={{ height: 16 }} />
        <div className="progress-track">
          <div className="progress-fill" style={{ width: `${progress}%` }} />
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 8 }}>
          <span className="muted">{statusLabel}</span>
          <strong>{progress}%</strong>
        </div>
        <div style={{ height: 16 }} />

        <div ref={logRef} className="log-console">
          {logs.length === 0 ? 'Logs will appear here when the run starts.' : logs.join('')}
        </div>
      </section>
    </div>
  )
}

function StepIcon({ label }: { label: string }) {
  return <strong>{label}</strong>
}

function formatStderr(text: string) {
  const isError = /(error|failed|traceback|exception|invalid value|does not exist)/i.test(text)
  return `${isError ? '[error]' : '[info]'} ${text}`
}
