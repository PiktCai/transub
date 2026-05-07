import { useEffect, useMemo, useState } from 'react'
import Card from '../components/Card'
import { Button, Input, Select } from '../components/FormControls'
import {
  LANGUAGES,
  PROVIDERS,
  TRANSCRIPTION_ENGINE,
  TRANSLATION_MODES,
  WHISPER_MODELS,
} from '../lib/constants'
import { readConfig, selectDirectory, writeConfig } from '../lib/bridge'

interface Props {
  configPath: string | null
}

type SetupTab = 'engine' | 'translate' | 'export'

export default function SetupPage({ configPath }: Props) {
  const [activeTab, setActiveTab] = useState<SetupTab>('engine')
  const [config, setConfig] = useState<Record<string, any>>({
    whisper: { model: 'base', language: 'auto', device: 'cpu', word_timestamps: true },
    llm: {
      provider: 'openai',
      model: 'gpt-5.4-mini',
      api_base: 'https://api.openai.com/v1',
      api_key_env: 'OPENAI_API_KEY',
      target_language: 'zh',
      batch_size: 5,
    },
    pipeline: { output_format: 'srt', audio_format: 'wav', output_dir: '' },
  })
  const [saved, setSaved] = useState(false)

  useEffect(() => {
    readConfig(configPath || undefined).then(existing => {
      if (existing) setConfig(previous => normalizeLegacyConfig({ ...previous, ...existing }))
    })
  }, [configPath])

  const update = (section: string, key: string, value: any) => {
    setConfig(previous => ({
      ...previous,
      [section]: { ...previous[section], [key]: value },
    }))
    setSaved(false)
  }

  const handleSave = async () => {
    await writeConfig(config, configPath || undefined)
    setSaved(true)
    window.setTimeout(() => setSaved(false), 1800)
  }

  const chooseOutputDir = async () => {
    const path = await selectDirectory()
    if (path) update('pipeline', 'output_dir', path)
  }

  const provider = config.llm?.provider || 'openai'
  const selectedProvider = PROVIDERS.find(item => item.id === provider) || PROVIDERS[0]
  const providerModels = useMemo(() => {
    const current = config.llm?.model || selectedProvider.defaultModel
    const models = selectedProvider.models.includes(current)
      ? selectedProvider.models
      : [current, ...selectedProvider.models]
    const options = models.filter(Boolean).map(model => ({ value: model, label: model }))
    return options.length > 0 ? options : [{ value: '', label: 'Paste model ID below' }]
  }, [config.llm?.model, selectedProvider])

  return (
    <div className="page">
      <section className="section-header">
        <div>
          <h1 className="page-title">Configuration</h1>
          <p className="page-kicker">Selectors first, escape hatches second. The defaults should run without reading docs.</p>
        </div>
        <div className="toolbar">
          <div className={`status-pill ${saved ? 'ready' : ''}`}>{saved ? 'Saved' : 'Unsaved'}</div>
          <Button variant="primary" onClick={handleSave}>{saved ? 'Saved' : 'Save config'}</Button>
        </div>
      </section>

      <div className="split-grid">
        <section className="surface">
          <div className="segmented" role="tablist" aria-label="Configuration sections">
            <button className={`segment ${activeTab === 'engine' ? 'is-active' : ''}`} onClick={() => setActiveTab('engine')}>Engine</button>
            <button className={`segment ${activeTab === 'translate' ? 'is-active' : ''}`} onClick={() => setActiveTab('translate')}>Translate</button>
            <button className={`segment ${activeTab === 'export' ? 'is-active' : ''}`} onClick={() => setActiveTab('export')}>Export</button>
          </div>

          <div style={{ height: 18 }} />

          {activeTab === 'engine' && (
            <div className="form-grid">
              <Select
                label="Model size"
                value={config.whisper?.model || WHISPER_MODELS[0]}
                options={WHISPER_MODELS.map(model => ({ value: model, label: model }))}
                onChange={value => update('whisper', 'model', value)}
                helper="base starts fast; large-v3-turbo is the high-quality default for longer work."
              />
              <Select
                label="Compute device"
                value={config.whisper?.device || 'cpu'}
                options={[
                  { value: 'cpu', label: 'CPU' },
                  { value: 'cuda', label: 'NVIDIA GPU / CUDA' },
                  { value: 'auto', label: 'Auto' },
                ]}
                onChange={value => update('whisper', 'device', value === 'auto' ? undefined : value)}
                helper="CPU works everywhere. CUDA is faster when available."
              />
              <Select
                label="Source language"
                value={config.whisper?.language || 'auto'}
                options={LANGUAGES}
                onChange={value => update('whisper', 'language', value)}
              />
              <Input
                label="Initial prompt"
                value={config.whisper?.initial_prompt || ''}
                onChange={value => update('whisper', 'initial_prompt', value)}
                placeholder="Optional terms, names, or style hints"
                helper="Useful for proper nouns and recurring terminology."
              />
            </div>
          )}

          {activeTab === 'translate' && (
            <div className="form-grid">
              <Select
                label="Translation mode"
                value={config.llm?.mode || 'llm'}
                options={TRANSLATION_MODES}
                onChange={value => update('llm', 'mode', value)}
              />
              <Select
                label="Provider"
                value={provider}
                options={PROVIDERS.map(item => ({ value: item.id, label: item.label }))}
                onChange={value => {
                  const nextProvider = PROVIDERS.find(item => item.id === value) || PROVIDERS[0]
                  update('llm', 'provider', value)
                  update('llm', 'model', nextProvider.defaultModel)
                  update('llm', 'api_base', nextProvider.defaultBase)
                  update('llm', 'api_key_env', nextProvider.keyEnv)
                }}
              />
              <Select
                label="LLM model"
                value={config.llm?.model || selectedProvider.defaultModel}
                options={providerModels}
                onChange={value => update('llm', 'model', value)}
                helper="Provider presets are editable later if needed."
              />
              <Input
                label="Custom model ID"
                value={config.llm?.model || selectedProvider.defaultModel}
                onChange={value => update('llm', 'model', value)}
                placeholder="Paste a model id from your provider"
                helper="Use this when a provider ships a new model before the app preset is updated."
              />
              <Select
                label="Target language"
                value={config.llm?.target_language || 'zh'}
                options={LANGUAGES.filter(language => language.value !== 'auto')}
                onChange={value => update('llm', 'target_language', value)}
              />
              <Select
                label="Batch size"
                value={String(config.llm?.batch_size || 5)}
                options={['1', '3', '5', '8', '12', '20'].map(value => ({ value, label: `${value} subtitles` }))}
                onChange={value => update('llm', 'batch_size', Number(value))}
              />
            </div>
          )}

          {activeTab === 'export' && (
            <div className="form-grid">
              <Select
                label="Subtitle format"
                value={config.pipeline?.output_format || 'srt'}
                options={[{ value: 'srt', label: 'SRT' }, { value: 'vtt', label: 'WebVTT' }]}
                onChange={value => update('pipeline', 'output_format', value)}
              />
              <Select
                label="Working audio format"
                value={config.pipeline?.audio_format || 'wav'}
                options={[
                  { value: 'wav', label: 'WAV' },
                  { value: 'flac', label: 'FLAC' },
                  { value: 'mp3', label: 'MP3' },
                  { value: 'm4a', label: 'M4A' },
                ]}
                onChange={value => update('pipeline', 'audio_format', value)}
              />
              <Input
                label="Output directory"
                value={config.pipeline?.output_dir || ''}
                onChange={value => update('pipeline', 'output_dir', value)}
                placeholder="Next to source video"
                helper="Use the chooser below to avoid path typos."
              />
              <div className="field">
                <span>Path chooser</span>
                <Button onClick={chooseOutputDir}>Choose folder</Button>
                <div className="helper-text">Leave empty to save next to the input video.</div>
              </div>
            </div>
          )}
        </section>

        <aside className="surface">
          <h2 className="section-title">Readiness</h2>
          <p className="page-kicker">This panel explains what the selected options mean before you run.</p>
          <div style={{ height: 16 }} />

          <div className="engine-list">
            <div className="engine-option is-active">
              <div className="option-head">
                <div className="option-title">{TRANSCRIPTION_ENGINE.label}</div>
                <div className="status-pill warning">{TRANSCRIPTION_ENGINE.short}</div>
              </div>
              <div className="option-description">{TRANSCRIPTION_ENGINE.description}</div>
              <div className="chip">uv sync</div>
            </div>

            <div className="engine-option">
              <div className="option-head">
                <div className="option-title">{selectedProvider.label}</div>
                <div className="status-pill">{selectedProvider.badge}</div>
              </div>
              <div className="option-description">
                Keys are stored per provider in <code>~/.transub/auth.toml</code>. Environment variables still win.
              </div>
              <div className="chip-row">
                <span className="chip">{selectedProvider.defaultBase}</span>
                <span className="chip">{selectedProvider.keyHint}</span>
              </div>
            </div>
          </div>
        </aside>
      </div>
    </div>
  )
}

function normalizeLegacyConfig(config: Record<string, any>) {
  const whisper = {
    model: normalizeWhisperModel(config.whisper?.model),
    device: config.whisper?.device || 'cpu',
    language: config.whisper?.language || 'auto',
    initial_prompt: config.whisper?.initial_prompt || '',
    temperature: config.whisper?.temperature ?? 0,
    compression_ratio_threshold: config.whisper?.compression_ratio_threshold ?? 2.6,
    logprob_threshold: config.whisper?.logprob_threshold ?? -1,
    no_speech_threshold: config.whisper?.no_speech_threshold ?? 0.3,
    condition_on_previous_text: config.whisper?.condition_on_previous_text ?? true,
    word_timestamps: true,
  }
  const provider = config.llm?.provider || 'openai'
  const selectedProvider = PROVIDERS.find(item => item.id === provider)
  if (!selectedProvider) return { ...config, whisper }

  const legacyModels = new Set(['gpt-4o-mini', 'gpt-4.1-mini', 'moonshot-v1-8k', 'qwen-turbo'])
  if (legacyModels.has(config.llm?.model)) {
    return {
      ...config,
      whisper,
      llm: {
        ...config.llm,
        model: selectedProvider.defaultModel,
        api_base: config.llm?.api_base || selectedProvider.defaultBase,
        api_key_env: config.llm?.api_key_env || selectedProvider.keyEnv,
      },
    }
  }

  return { ...config, whisper }
}

function normalizeWhisperModel(model: string | undefined) {
  if (!model) return 'base'
  if (WHISPER_MODELS.includes(model)) return model
  return 'base'
}
