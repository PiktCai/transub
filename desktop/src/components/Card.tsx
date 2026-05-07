import { Button } from './FormControls'

interface CardProps {
  title: string
  description?: string
  icon?: React.ReactNode
  status?: 'ready' | 'warning' | 'error' | 'pending'
  action?: { label: string; onClick: () => void; variant?: 'primary' | 'secondary' | 'accent' }
  children?: React.ReactNode
}

export default function Card({ title, description, icon, status, action, children }: CardProps) {
  return (
    <section className="card">
      <div className="card-header">
        <div className="card-icon">{icon || <DefaultIcon />}</div>
        <div>
          <h3 className="card-title">{title}</h3>
          {description && <div className="card-description">{description}</div>}
        </div>
        {status && <span className={`card-status ${status}`} />}
      </div>
      {children}
      {action && (
        <div>
          <Button size="sm" variant={action.variant || 'secondary'} onClick={action.onClick}>
            {action.label}
          </Button>
        </div>
      )}
    </section>
  )
}

function DefaultIcon() {
  return (
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path d="M5 7.5h14M5 12h14M5 16.5h10" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
    </svg>
  )
}
