import { Heart, Thermometer, Activity, Droplets } from 'lucide-react'
import { fmt, fmtInt, activityLabel } from '../utils/formatters'

function MetricRow({ icon: Icon, label, value, unit, color, warning }) {
  return (
    <div className="flex items-center justify-between py-2.5 border-b last:border-0" style={{ borderColor: 'var(--border)' }}>
      <div className="flex items-center gap-2.5">
        <div
          className="w-7 h-7 rounded-md flex items-center justify-center shrink-0"
          style={{ background: `color-mix(in srgb, ${color} 15%, transparent)` }}
        >
          <Icon size={14} style={{ color }} />
        </div>
        <span className="text-sm" style={{ color: 'var(--text-secondary)' }}>{label}</span>
      </div>
      <div className="text-right">
        {value !== null && value !== undefined ? (
          <span
            className="font-mono text-sm font-semibold"
            style={{ color: warning ? 'var(--accent-warning)' : 'var(--text-primary)' }}
          >
            {value}<span className="text-xs font-normal ml-0.5" style={{ color: 'var(--text-muted)' }}>{unit}</span>
          </span>
        ) : (
          <span className="text-sm font-mono" style={{ color: 'var(--text-muted)' }}>—</span>
        )}
      </div>
    </div>
  )
}

export default function HealthCard({ reading }) {
  const hr   = reading?.heart_rate
  const spo2 = reading?.spo2
  const temp = reading?.body_temperature
  const act  = reading?.activity_level
  const eda  = reading?.eda

  return (
    <div className="card">
      <h3 className="text-xs font-semibold uppercase tracking-wider mb-3" style={{ color: 'var(--text-muted)' }}>
        Physiological
      </h3>

      <MetricRow
        icon={Heart}
        label="Heart Rate"
        value={hr != null ? fmtInt(hr) : null}
        unit="bpm"
        color="var(--accent-danger)"
        warning={hr != null && (hr > 100 || hr < 50)}
      />
      <MetricRow
        icon={Droplets}
        label="SpO₂"
        value={spo2 != null ? fmt(spo2, 1) : null}
        unit="%"
        color="var(--accent-env)"
        warning={spo2 != null && spo2 < 95}
      />
      <MetricRow
        icon={Thermometer}
        label="Body Temp"
        value={temp != null ? fmt(temp, 1) : null}
        unit="°C"
        color="var(--accent-warning)"
        warning={temp != null && (temp > 38 || temp < 35)}
      />
      <MetricRow
        icon={Activity}
        label="Activity"
        value={act != null ? activityLabel(act) : null}
        unit=""
        color="var(--accent-primary)"
        warning={false}
      />
      {eda != null && (
        <MetricRow
          icon={Activity}
          label="EDA"
          value={fmt(eda, 2)}
          unit="µS"
          color="var(--accent-health)"
          warning={false}
        />
      )}
    </div>
  )
}
