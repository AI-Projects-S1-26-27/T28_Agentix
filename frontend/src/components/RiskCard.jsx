import { ShieldCheck, ShieldAlert, ShieldX, Shield } from 'lucide-react'
import { getRiskMeta, fmtRelative } from '../utils/formatters'

const ICONS = {
  low:       ShieldCheck,
  moderate:  Shield,
  high:      ShieldAlert,
  very_high: ShieldAlert,
  severe:    ShieldX,
  unknown:   Shield,
}

export default function RiskCard({ risk }) {
  const level = risk?.risk_level ?? 'unknown'
  const meta  = getRiskMeta(level)
  const Icon  = ICONS[level] ?? Shield

  const signals = risk?.contributing_signals ?? []
  const unavail = risk?.unavailable_inputs ?? []

  return (
    <div
      className="card"
      style={{ borderColor: `color-mix(in srgb, ${meta.color} 25%, var(--border))` }}
    >
      {/* Header */}
      <div className="flex items-start justify-between mb-3">
        <div>
          <p className="text-xs font-semibold uppercase tracking-wider mb-1" style={{ color: 'var(--text-muted)' }}>
            Risk Level
          </p>
          <div className="flex items-center gap-2">
            <Icon size={18} style={{ color: meta.color }} strokeWidth={2} />
            <span className="text-xl font-bold" style={{ color: meta.color }}>
              {meta.label}
            </span>
          </div>
        </div>

        {/* Severity bar */}
        <div className="flex flex-col items-end gap-1">
          <span className="text-xs" style={{ color: 'var(--text-muted)' }}>Severity</span>
          <div className="flex gap-1">
            {[0,1,2,3,4].map(i => (
              <div
                key={i}
                className="w-3 h-3 rounded-sm"
                style={{
                  background: i <= (risk?.severity ?? -1) ? meta.color : 'var(--surface-elevated)',
                  opacity: i <= (risk?.severity ?? -1) ? 1 : 0.4,
                }}
              />
            ))}
          </div>
        </div>
      </div>

      {/* Explanation */}
      {risk?.explanation && (
        <p className="text-xs leading-relaxed mb-3 p-2.5 rounded-md" style={{ background: 'var(--surface-elevated)', color: 'var(--text-secondary)' }}>
          {risk.explanation}
        </p>
      )}

      {/* Contributing signals */}
      {signals.length > 0 && (
        <div className="mb-2">
          <p className="text-xs mb-1.5 font-medium" style={{ color: 'var(--text-muted)' }}>Contributing</p>
          <div className="flex flex-wrap gap-1.5">
            {signals.map(s => (
              <span key={s} className="badge text-xs" style={{ background: `color-mix(in srgb, ${meta.color} 12%, var(--surface-elevated))`, color: meta.color }}>
                {s.replace(/_/g, ' ')}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Unavailable */}
      {unavail.length > 0 && (
        <div>
          <p className="text-xs mb-1.5 font-medium" style={{ color: 'var(--text-muted)' }}>Unavailable inputs</p>
          <div className="flex flex-wrap gap-1.5">
            {unavail.map(s => (
              <span key={s} className="badge text-xs" style={{ background: 'var(--surface-elevated)', color: 'var(--text-muted)' }}>
                {s.replace(/_/g, ' ')}
              </span>
            ))}
          </div>
        </div>
      )}

      {risk?.timestamp && (
        <p className="text-xs mt-3 pt-2 border-t" style={{ borderColor: 'var(--border)', color: 'var(--text-muted)' }}>
          Updated {fmtRelative(risk.timestamp)}
        </p>
      )}
    </div>
  )
}
