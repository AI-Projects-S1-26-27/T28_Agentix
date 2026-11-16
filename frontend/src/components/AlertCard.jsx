import { AlertTriangle, Info, Siren, CheckCircle } from 'lucide-react'
import { fmtRelative } from '../utils/formatters'

const LEVEL_META = {
  info:     { Icon: Info,          color: 'var(--accent-primary)',  bg: 'color-mix(in srgb, var(--accent-primary) 10%, var(--surface))' },
  warning:  { Icon: AlertTriangle, color: 'var(--accent-warning)',  bg: 'color-mix(in srgb, var(--accent-warning) 10%, var(--surface))' },
  critical: { Icon: Siren,         color: 'var(--accent-critical)', bg: 'color-mix(in srgb, var(--accent-critical) 10%, var(--surface))' },
}

export default function AlertCard({ alert, onAcknowledge }) {
  const { Icon, color, bg } = LEVEL_META[alert?.level] ?? LEVEL_META.info

  return (
    <div
      className="rounded-lg border p-4 flex gap-3"
      style={{
        background: alert?.acknowledged ? 'var(--surface)' : bg,
        borderColor: alert?.acknowledged ? 'var(--border)' : `color-mix(in srgb, ${color} 30%, var(--border))`,
        opacity: alert?.acknowledged ? 0.65 : 1,
      }}
    >
      <div className="shrink-0 mt-0.5">
        <Icon size={16} style={{ color: alert?.acknowledged ? 'var(--text-muted)' : color }} />
      </div>

      <div className="flex-1 min-w-0">
        <div className="flex items-start justify-between gap-2 mb-0.5">
          <p className="text-sm font-semibold leading-snug" style={{ color: alert?.acknowledged ? 'var(--text-secondary)' : 'var(--text-primary)' }}>
            {alert?.message}
          </p>
          {!alert?.acknowledged && onAcknowledge && (
            <button
              onClick={() => onAcknowledge(alert.id)}
              className="shrink-0 flex items-center gap-1 text-xs px-2 py-1 rounded-md border transition-colors"
              style={{ borderColor: 'var(--border)', color: 'var(--text-muted)' }}
              onMouseEnter={e => { e.currentTarget.style.background = 'var(--surface-elevated)'; e.currentTarget.style.color = 'var(--text-primary)' }}
              onMouseLeave={e => { e.currentTarget.style.background = 'transparent'; e.currentTarget.style.color = 'var(--text-muted)' }}
            >
              <CheckCircle size={12} />
              Ack
            </button>
          )}
        </div>

        {alert?.explanation && (
          <p className="text-xs leading-relaxed mb-1.5" style={{ color: 'var(--text-secondary)' }}>
            {alert.explanation}
          </p>
        )}

        <div className="flex items-center gap-3 flex-wrap">
          {alert?.category && (
            <span
              className="badge text-xs"
              style={{ background: `color-mix(in srgb, ${color} 12%, transparent)`, color }}
            >
              {alert.category.replace(/_/g, ' ')}
            </span>
          )}
          <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
            {fmtRelative(alert?.timestamp)}
          </span>
          {alert?.acknowledged && (
            <span className="text-xs" style={{ color: 'var(--accent-health)' }}>✓ Acknowledged</span>
          )}
        </div>
      </div>
    </div>
  )
}
