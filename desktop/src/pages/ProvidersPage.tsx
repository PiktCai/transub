import { useEffect, useMemo, useState } from 'react'
import { Button, Input, Select } from '../components/FormControls'
import { PROVIDERS } from '../lib/constants'
import { clearProviderAuth, readAuth, saveProviderAuth } from '../lib/bridge'

export default function ProvidersPage() {
  const [auth, setAuth] = useState<Record<string, { has_key: boolean; api_base: string }>>({})
  const [activeProvider, setActiveProvider] = useState(PROVIDERS[0].id)
  const [draftKey, setDraftKey] = useState('')
  const [draftBase, setDraftBase] = useState('')
  const [draftModel, setDraftModel] = useState(PROVIDERS[0].defaultModel)
  const [saved, setSaved] = useState(false)

  const provider = useMemo(
    () => PROVIDERS.find(item => item.id === activeProvider) || PROVIDERS[0],
    [activeProvider],
  )
  const modelOptions = useMemo(() => {
    const models = provider.models.includes(draftModel) ? provider.models : [draftModel, ...provider.models]
    const options = models.filter(Boolean).map(model => ({ value: model, label: model }))
    return options.length > 0 ? options : [{ value: '', label: 'Paste model ID below' }]
  }, [draftModel, provider.models])

  useEffect(() => {
    readAuth().then(existing => {
      setAuth(existing)
      setDraftBase(provider.defaultBase)
      setDraftModel(provider.defaultModel)
    })
  }, [])

  useEffect(() => {
    setDraftBase(provider.defaultBase)
    setDraftModel(provider.defaultModel)
    setSaved(false)
  }, [activeProvider, provider.defaultBase, provider.defaultModel])

  const handleSave = async () => {
    await saveProviderAuth(activeProvider, {
      api_key: draftKey || undefined,
      api_base: draftBase || undefined,
    })
    setAuth(previous => ({
      ...previous,
      [activeProvider]: { has_key: !!draftKey, api_base: draftBase || '' },
    }))
    setSaved(true)
    window.setTimeout(() => setSaved(false), 1600)
  }

  const handleClear = async () => {
    await clearProviderAuth(activeProvider)
    setAuth(previous => {
      const next = { ...previous }
      delete next[activeProvider]
      return next
    })
    setDraftKey('')
    setDraftBase(provider.defaultBase)
    setDraftModel(provider.defaultModel)
  }

  return (
    <div className="page">
      <section className="section-header">
        <div>
          <h1 className="page-title">Provider Studio</h1>
          <p className="page-kicker">
            One provider, one credential block. Transub stores this locally in <code>~/.transub/auth.toml</code>;
            environment variables remain higher priority.
          </p>
        </div>
        <div className={`status-pill ${saved ? 'ready' : auth[activeProvider]?.has_key ? 'ready' : 'warning'}`}>
          {saved ? 'Saved' : auth[activeProvider]?.has_key ? 'Key saved' : 'Missing key'}
        </div>
      </section>

      <div className="split-grid">
        <section className="surface">
          <div className="provider-list">
            {PROVIDERS.map(item => {
              const hasKey = !!auth[item.id]?.has_key
              const active = item.id === activeProvider
              return (
                <button
                  key={item.id}
                  className={`provider-row ${active ? 'is-active' : ''}`}
                  onClick={() => setActiveProvider(item.id)}
                >
                  <div className="provider-head">
                    <div>
                      <div className="provider-title">{item.label}</div>
                      <div className="provider-meta">{item.defaultModel}</div>
                    </div>
                    <div className={`status-pill ${hasKey || item.id === 'ollama' ? 'ready' : ''}`}>
                      {hasKey ? 'Saved' : item.id === 'ollama' ? 'Local' : item.badge}
                    </div>
                  </div>
                  <div className="provider-meta">{item.defaultBase}</div>
                </button>
              )
            })}
          </div>
        </section>

        <section className="surface">
          <div className="section-header">
            <div>
              <h2 className="section-title">{provider.label}</h2>
              <p className="page-kicker">{provider.badge} provider preset</p>
            </div>
            <div className="status-pill">{provider.keyHint}</div>
          </div>

          <div style={{ height: 18 }} />

          <div className="form-grid">
            <Select
              label="Default model"
              value={draftModel}
              options={modelOptions}
              onChange={setDraftModel}
              helper="Used as the suggested translation model in Setup."
            />
            <Input
              label="API base"
              value={draftBase}
              onChange={setDraftBase}
              placeholder={provider.defaultBase}
              helper="Compatible providers usually end with /v1."
            />
          </div>

          <div style={{ height: 14 }} />

          <Input
            label={provider.id === 'ollama' ? 'API key optional' : 'API key'}
            value={draftKey}
            onChange={setDraftKey}
            type="password"
            placeholder={provider.id === 'ollama' ? 'Usually empty for local Ollama' : 'Paste provider key'}
            helper="The desktop app writes this to the local auth file; it is not committed to the repo."
          />

          <div style={{ height: 14 }} />

          <Input
            label="Custom model ID"
            value={draftModel}
            onChange={setDraftModel}
            placeholder="Add a provider model id"
            helper="Like Cherry Studio, presets are just a starting catalog. You can paste a fresh model id here."
          />

          <div style={{ height: 18 }} />

          <div className="button-row">
            <Button variant="primary" onClick={handleSave}>Save provider</Button>
            <Button variant="danger" onClick={handleClear}>Clear</Button>
            <Button variant="ghost" onClick={() => {
              setDraftBase(provider.defaultBase)
              setDraftModel(provider.defaultModel)
            }}>
              Reset preset
            </Button>
          </div>

          <div style={{ height: 18 }} />

          <div className="chip-row">
            <span className="chip">Auth file: ~/.transub/auth.toml</span>
            <span className="chip">Env wins over file</span>
            <span className="chip">Provider scoped</span>
          </div>
        </section>
      </div>
    </div>
  )
}
