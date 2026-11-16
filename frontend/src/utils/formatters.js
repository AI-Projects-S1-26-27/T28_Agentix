// ─── Number & unit formatters ───────────────────────────────────────────────

export function fmt(value, decimals = 1, fallback = '—') {
  if (value === null || value === undefined || Number.isNaN(value)) return fallback
  return Number(value).toFixed(decimals)
}

export function fmtInt(value, fallback = '—') {
  if (value === null || value === undefined) return fallback
  return Math.round(value).toString()
}

// ─── Time formatters ─────────────────────────────────────────────────────────

export function fmtTime(ts) {
  if (!ts) return '—'
  const d = new Date(ts)
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
}

export function fmtDate(ts) {
  if (!ts) return '—'
  const d = new Date(ts)
  return d.toLocaleDateString([], { month: 'short', day: 'numeric' })
}

export function fmtDateTime(ts) {
  if (!ts) return '—'
  const d = new Date(ts)
  return d.toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })
}

export function fmtRelative(ts) {
  if (!ts) return '—'
  const diff = (Date.now() - new Date(ts).getTime()) / 1000
  if (diff < 5) return 'just now'
  if (diff < 60) return `${Math.floor(diff)}s ago`
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`
  return fmtDate(ts)
}

// ─── Risk level helpers ───────────────────────────────────────────────────────

export const RISK_META = {
  low:       { label: 'Low',       color: 'var(--accent-health)',   bg: 'color-mix(in srgb, var(--accent-health) 12%, transparent)',   severity: 0 },
  moderate:  { label: 'Moderate',  color: 'var(--accent-warning)',  bg: 'color-mix(in srgb, var(--accent-warning) 12%, transparent)',  severity: 1 },
  high:      { label: 'High',      color: 'var(--accent-danger)',   bg: 'color-mix(in srgb, var(--accent-danger) 12%, transparent)',   severity: 2 },
  very_high: { label: 'Very High', color: 'var(--accent-danger)',   bg: 'color-mix(in srgb, var(--accent-danger) 15%, transparent)',   severity: 3 },
  severe:    { label: 'Severe',    color: 'var(--accent-critical)', bg: 'color-mix(in srgb, var(--accent-critical) 15%, transparent)', severity: 4 },
  unknown:   { label: 'Unknown',   color: 'var(--text-muted)',      bg: 'var(--surface-elevated)',                                      severity: -1 },
}

export function getRiskMeta(level) {
  return RISK_META[level] ?? RISK_META.unknown
}

// ─── Alert level helpers ──────────────────────────────────────────────────────

export const ALERT_META = {
  info:     { label: 'Info',     color: 'var(--accent-primary)', icon: 'ℹ' },
  warning:  { label: 'Warning',  color: 'var(--accent-warning)', icon: '⚠' },
  critical: { label: 'Critical', color: 'var(--accent-critical)', icon: '🚨' },
}

export function getAlertMeta(level) {
  return ALERT_META[level] ?? ALERT_META.info
}

// ─── Signal label map ─────────────────────────────────────────────────────────

export const SIGNAL_LABELS = {
  heart_rate:          { label: 'Heart Rate',   unit: 'bpm',  color: 'var(--accent-danger)' },
  spo2:                { label: 'SpO₂',          unit: '%',    color: 'var(--accent-env)' },
  body_temperature:    { label: 'Body Temp',    unit: '°C',   color: 'var(--accent-warning)' },
  activity_level:      { label: 'Activity',     unit: '',     color: 'var(--accent-primary)' },
  ambient_temperature: { label: 'Ambient Temp', unit: '°C',   color: 'var(--accent-warning)' },
  humidity:            { label: 'Humidity',     unit: '%',    color: 'var(--accent-env)' },
  aqi:                 { label: 'AQI',          unit: '',     color: 'var(--accent-critical)' },
  pm25:                { label: 'PM2.5',        unit: 'µg/m³', color: 'var(--accent-critical)' },
  eda:                 { label: 'EDA',          unit: 'µS',   color: 'var(--accent-health)' },
}

export function getSignalMeta(key) {
  return SIGNAL_LABELS[key] ?? { label: key, unit: '', color: 'var(--accent-primary)' }
}

// ─── AQI category helper ──────────────────────────────────────────────────────

export function aqiCategory(aqi) {
  if (aqi == null) return { label: 'N/A', color: 'var(--text-muted)' }
  if (aqi <= 50)  return { label: 'Good',                    color: 'var(--accent-health)' }
  if (aqi <= 100) return { label: 'Moderate',                color: 'var(--accent-warning)' }
  if (aqi <= 150) return { label: 'Unhealthy (Sensitive)',   color: '#f97316' }
  if (aqi <= 200) return { label: 'Unhealthy',               color: 'var(--accent-danger)' }
  if (aqi <= 300) return { label: 'Very Unhealthy',          color: 'var(--accent-critical)' }
  return                  { label: 'Hazardous',              color: '#7c3aed' }
}

// ─── Activity label ───────────────────────────────────────────────────────────

export function activityLabel(val) {
  if (val == null) return '—'
  if (val < 0.2)  return 'Resting'
  if (val < 0.5)  return 'Light'
  if (val < 0.75) return 'Moderate'
  return 'Active'
}
