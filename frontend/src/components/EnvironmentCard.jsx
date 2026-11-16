import { Thermometer, Droplets, Wind, CloudFog } from 'lucide-react'
import { fmt, fmtInt, aqiCategory } from '../utils/formatters'

function MetricRow({ icon: Icon, label, value, unit, color, warning, badge }) {
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
      <div className="flex items-center gap-2">
        {badge && (
          <span
            className="text-xs px-1.5 py-0.5 rounded font-medium"
            style={{ background: `color-mix(in srgb, ${badge.color} 15%, transparent)`, color: badge.color }}
          >
            {badge.label}
          </span>
        )}
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

export default function EnvironmentCard({ reading }) {
  const ambTemp = reading?.ambient_temperature
  const hum     = reading?.humidity
  const aqi     = reading?.aqi
  const pm25    = reading?.pm25

  const aqiMeta = aqiCategory(aqi)

  return (
    <div className="card">
      <h3 className="text-xs font-semibold uppercase tracking-wider mb-3" style={{ color: 'var(--text-muted)' }}>
        Environment
      </h3>

      <MetricRow
        icon={Thermometer}
        label="Ambient Temp"
        value={ambTemp != null ? fmt(ambTemp, 1) : null}
        unit="°C"
        color="var(--accent-warning)"
        warning={ambTemp != null && ambTemp > 35}
      />
      <MetricRow
        icon={Droplets}
        label="Humidity"
        value={hum != null ? fmtInt(hum) : null}
        unit="%"
        color="var(--accent-env)"
        warning={hum != null && hum > 80}
      />
      <MetricRow
        icon={Wind}
        label="AQI"
        value={aqi != null ? fmtInt(aqi) : null}
        unit=""
        color="var(--accent-critical)"
        warning={aqi != null && aqi > 100}
        badge={aqi != null ? aqiMeta : undefined}
      />
      <MetricRow
        icon={CloudFog}
        label="PM2.5"
        value={pm25 != null ? fmt(pm25, 1) : null}
        unit="µg/m³"
        color="var(--accent-critical)"
        warning={pm25 != null && pm25 > 35}
      />
    </div>
  )
}
