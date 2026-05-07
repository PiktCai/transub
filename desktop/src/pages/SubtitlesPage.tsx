import { useEffect, useState } from 'react'
import { Button } from '../components/FormControls'
import { bridge } from '../lib/bridge'
import type { AppState } from '../App'

interface Props {
  state: AppState
}

interface SubtitleEntry {
  index: number
  start: string
  end: string
  text: string
}

function parseSrt(content: string): SubtitleEntry[] {
  const entries: SubtitleEntry[] = []
  const blocks = content.trim().split(/\n\s*\n/)
  for (const block of blocks) {
    const lines = block.trim().split('\n')
    if (lines.length < 3) continue
    const index = Number.parseInt(lines[0], 10)
    const timeMatch = lines[1].match(/([\d:,.]+)\s*-->\s*([\d:,.]+)/)
    if (!timeMatch) continue
    entries.push({ index, start: timeMatch[1], end: timeMatch[2], text: lines.slice(2).join('\n') })
  }
  return entries
}

function parseVtt(content: string): SubtitleEntry[] {
  const lines = content.split('\n')
  const entries: SubtitleEntry[] = []
  let i = lines[0]?.startsWith('WEBVTT') ? 1 : 0
  while (i < lines.length) {
    const line = lines[i].trim()
    if (!line) {
      i += 1
      continue
    }
    const timeMatch = line.match(/([\d:,.]+)\s*-->\s*([\d:,.]+)/)
    if (timeMatch) {
      const textLines: string[] = []
      i += 1
      while (i < lines.length && lines[i].trim()) {
        textLines.push(lines[i].trim())
        i += 1
      }
      entries.push({ index: entries.length + 1, start: timeMatch[1], end: timeMatch[2], text: textLines.join('\n') })
    } else {
      i += 1
    }
  }
  return entries
}

export default function SubtitlesPage({ state }: Props) {
  const [entries, setEntries] = useState<SubtitleEntry[]>([])
  const [source, setSource] = useState<string | null>(null)

  useEffect(() => {
    if (!state.subtitleContent) return
    const parsed = state.subtitleFormat === 'vtt' ? parseVtt(state.subtitleContent) : parseSrt(state.subtitleContent)
    setEntries(parsed)
    setSource('pipeline output')
  }, [state.subtitleContent, state.subtitleFormat])

  const handleLoadFile = async () => {
    const path = await bridge.dialog.openFile({
      filters: [
        { name: 'Subtitle Files', extensions: ['srt', 'vtt'] },
        { name: 'All Files', extensions: ['*'] },
      ],
    })
    if (!path) return
    const result = await bridge.fs.readFile(path)
    if (!result.success || !result.content) return
    const fmt = path.endsWith('.vtt') ? 'vtt' : 'srt'
    setEntries(fmt === 'vtt' ? parseVtt(result.content) : parseSrt(result.content))
    setSource(path)
  }

  return (
    <div className="page">
      <section className="section-header">
        <div>
          <h1 className="page-title">Subtitle Preview</h1>
          <p className="page-kicker">Load an existing subtitle file or inspect the latest pipeline output.</p>
        </div>
        <div className="toolbar">
          <div className="status-pill">{entries.length} lines</div>
          <Button variant="primary" onClick={handleLoadFile}>Load SRT/VTT</Button>
        </div>
      </section>

      <section className="subtitle-panel">
        {entries.length === 0 ? (
          <div className="empty-state">
            <div>
              <div style={{ fontSize: 42, marginBottom: 10 }}>▭</div>
              <strong>No subtitles loaded</strong>
              <div className="helper-text">Run a pipeline or load a .srt/.vtt file to preview captions here.</div>
            </div>
          </div>
        ) : (
          <>
            <div className="section-header" style={{ marginBottom: 14 }}>
              <div className="helper-text">{source}</div>
              <div className="chip-row">
                <span className="chip">SRT/VTT parser</span>
                <span className="chip">Read-only preview</span>
              </div>
            </div>
            <table className="subtitle-table">
              <thead>
                <tr>
                  <th style={{ width: 70 }}>#</th>
                  <th style={{ width: 150 }}>Start</th>
                  <th style={{ width: 150 }}>End</th>
                  <th>Text</th>
                </tr>
              </thead>
              <tbody>
                {entries.map(entry => (
                  <tr key={`${entry.index}-${entry.start}`}>
                    <td>{entry.index}</td>
                    <td><code>{entry.start}</code></td>
                    <td><code>{entry.end}</code></td>
                    <td>{entry.text}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </>
        )}
      </section>
    </div>
  )
}
