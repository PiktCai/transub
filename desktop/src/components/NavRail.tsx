import { NAV_ITEMS } from '../lib/constants'

interface NavRailProps {
  current: string
  onNavigate: (path: string) => void
}

export default function NavRail({ current, onNavigate }: NavRailProps) {
  return (
    <nav className="nav-rail" aria-label="Primary navigation">
      {NAV_ITEMS.map(item => {
        const active = item.path === current
        return (
          <button
            key={item.path}
            className={`nav-item ${active ? 'is-active' : ''}`}
            onClick={() => onNavigate(item.path)}
            title={item.hint}
          >
            <span className="nav-icon"><NavIcon name={item.icon} /></span>
            <span className="nav-label">{item.label}</span>
          </button>
        )
      })}
      <div className="nav-version">transub desktop</div>
    </nav>
  )
}

function NavIcon({ name }: { name: string }) {
  const common = {
    width: 24,
    height: 24,
    viewBox: '0 0 24 24',
    fill: 'none',
    'aria-hidden': true,
  }

  if (name === 'rocket') {
    return (
      <svg {...common}>
        <path d="M8.5 15.5 5 19l.8-4.8L4 12.4l4.8-.8L12.3 8C14.2 6.1 16.5 4.9 19 4.5c-.4 2.5-1.6 4.8-3.5 6.7l-3.6 3.6-.8 4.8-1.8-1.8-4.8.8 3.5-3.1Z" stroke="currentColor" strokeWidth="1.9" strokeLinejoin="round" />
        <path d="M14.2 8.8h.01" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
      </svg>
    )
  }

  if (name === 'sliders') {
    return (
      <svg {...common}>
        <path d="M5 7h14M5 17h14M8 4v6M16 14v6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
        <path d="M8 10a3 3 0 1 0 0-6 3 3 0 0 0 0 6ZM16 20a3 3 0 1 0 0-6 3 3 0 0 0 0 6Z" stroke="currentColor" strokeWidth="2" />
      </svg>
    )
  }

  if (name === 'key') {
    return (
      <svg {...common}>
        <path d="M9.5 14.5a4.5 4.5 0 1 1 3.6-7.2 4.5 4.5 0 0 1-3.6 7.2Z" stroke="currentColor" strokeWidth="2" />
        <path d="m13 13 6 6M17 17l2-2M15 15l2-2" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
      </svg>
    )
  }

  if (name === 'play') {
    return (
      <svg {...common}>
        <path d="M8 5.8v12.4c0 .9 1 1.4 1.7.9l8.6-6.2c.6-.4.6-1.3 0-1.8L9.7 4.9C9 4.4 8 4.9 8 5.8Z" fill="currentColor" />
      </svg>
    )
  }

  return (
    <svg {...common}>
      <path d="M5 6.5h14v11H5zM8 10h8M8 13.5h5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  )
}
