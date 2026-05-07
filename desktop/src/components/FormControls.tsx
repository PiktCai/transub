interface SelectProps {
  label: string
  value: string
  options: { value: string; label: string }[]
  onChange: (value: string) => void
  helper?: string
}

export function Select({ label, value, options, onChange, helper }: SelectProps) {
  return (
    <label className="field">
      <span>{label}</span>
      <select className="select" value={value} onChange={event => onChange(event.target.value)}>
        {options.map(option => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
      {helper && <div className="helper-text">{helper}</div>}
    </label>
  )
}

interface InputProps {
  label: string
  value: string
  onChange: (value: string) => void
  placeholder?: string
  type?: string
  disabled?: boolean
  helper?: string
}

export function Input({ label, value, onChange, placeholder, type = 'text', disabled, helper }: InputProps) {
  return (
    <label className="field">
      <span>{label}</span>
      <input
        className="input"
        type={type}
        value={value}
        onChange={event => onChange(event.target.value)}
        placeholder={placeholder}
        disabled={disabled}
      />
      {helper && <div className="helper-text">{helper}</div>}
    </label>
  )
}

interface ButtonProps {
  children: React.ReactNode
  onClick?: () => void
  variant?: 'primary' | 'secondary' | 'danger' | 'ghost' | 'accent'
  disabled?: boolean
  size?: 'sm' | 'md'
  type?: 'button' | 'submit'
}

export function Button({
  children,
  onClick,
  variant = 'secondary',
  disabled,
  size = 'md',
  type = 'button',
}: ButtonProps) {
  const variantClass = variant === 'secondary' ? '' : variant
  return (
    <button
      className={['button', variantClass, size === 'sm' ? 'sm' : ''].filter(Boolean).join(' ')}
      type={type}
      onClick={onClick}
      disabled={disabled}
    >
      {children}
    </button>
  )
}
