import { Routes, Route, useNavigate, useLocation } from 'react-router-dom'
import { useState, useCallback } from 'react'
import StartPage from './pages/StartPage'
import SetupPage from './pages/SetupPage'
import ProvidersPage from './pages/ProvidersPage'
import RunPage from './pages/RunPage'
import SubtitlesPage from './pages/SubtitlesPage'
import CommandBar from './components/CommandBar'
import NavRail from './components/NavRail'
import { NAV_ITEMS } from './lib/constants'

export interface AppState {
  videoPath: string | null
  configPath: string | null
  isRunning: boolean
  subtitleContent: string | null
  subtitleFormat: 'srt' | 'vtt'
  logs: string[]
}

export default function App() {
  const navigate = useNavigate()
  const location = useLocation()
  const [state, setState] = useState<AppState>({
    videoPath: null,
    configPath: null,
    isRunning: false,
    subtitleContent: null,
    subtitleFormat: 'srt',
    logs: [],
  })

  const updateState = useCallback((patch: Partial<AppState>) => {
    setState(prev => ({ ...prev, ...patch }))
  }, [])

  const currentRoute = NAV_ITEMS.find(item =>
    item.path === '/' ? location.pathname === '/' : location.pathname.startsWith(item.path)
  )?.path || '/'

  return (
    <div className="app-shell">
      <CommandBar
        videoPath={state.videoPath}
        isRunning={state.isRunning}
        onVideoSelect={(path) => updateState({ videoPath: path })}
        onRun={() => navigate('/run')}
        onTranscribeOnly={() => navigate('/run', { state: { transcribeOnly: true } })}
      />
      <div className="app-body">
        <NavRail current={currentRoute} onNavigate={navigate} />
        <main className="app-main">
          <Routes>
            <Route path="/" element={<StartPage state={state} updateState={updateState} />} />
            <Route path="/setup" element={<SetupPage configPath={state.configPath} />} />
            <Route path="/providers" element={<ProvidersPage />} />
            <Route path="/run" element={<RunPage state={state} updateState={updateState} />} />
            <Route path="/subtitles" element={<SubtitlesPage state={state} />} />
          </Routes>
        </main>
      </div>
    </div>
  )
}
