import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, ReferenceLine } from 'recharts'
import { SIGNAL_LABELS, fmt, fmtTime } from '../utils/formatters'

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  return (
    <div className="card-elevated rounded-lg px-3 py-2 text-xs shadow-lg" style={{ border: '1px solid var(--border)' }}>
      <p className="mb-1" style={{ color: 'var(--text-muted)' }}>{fmtTime(label)}</p>
      {payload.map(p => (
        <div key={p.dataKey} className="flex items-center gap-2">
          <span style={{ color: p.color }}>●</span>
          <span style={{ color: 'var(--text-secondary)' }}>{p.name}:</span>
          <span className="font-semibold" style={{ color: 'var(--text-primary)' }}>{fmt(p.value)}</span>
        </div>
      ))}
    </div>
  )
}

export default function LiveChart({ signal, data = [], refLines = [] }) {
  const meta = SIGNAL_LABELS[signal] ?? { label: signal, unit: '', color: 'var(--accent-primary)' }
  const values = data.map(d => d[signal]).filter(v => v != null)
  const min = values.length ? Math.min(...values) : 0
  const max = values.length ? Math.max(...values) : 100
  const pad = (max - min) * 0.15 || 5

  return (
    <div className="card rounded-xl p-4" style={{ minHeight: 180 }}>
      <div className="flex items-center justify-between mb-3">
        <span className="text-sm font-semibold" style={{ color: 'var(--text-secondary)' }}>
          {meta.label}
        </span>
        <span className="badge text-xs" style={{ background: `color-mix(in srgb, ${meta.color} 12%, transparent)`, color: meta.color }}>
          {meta.unit}
        </span>
      </div>

      <ResponsiveContainer width="100%" height={120}>
        <LineChart data={data} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
          <XAxis
            dataKey="timestamp"
            tickFormatter={fmtTime}
            tick={{ fontSize: 10, fill: 'var(--text-muted)' }}
            axisLine={false}
            tickLine={false}
            interval="preserveStartEnd"
          />
          <YAxis
            domain={[min - pad, max + pad]}
            tick={{ fontSize: 10, fill: 'var(--text-muted)' }}
            axisLine={false}
            tickLine={false}
            tickFormatter={v => fmt(v, 0)}
          />
          <Tooltip content={<CustomTooltip />} />
          {refLines.map((r, i) => (
            <ReferenceLine key={i} y={r.value} stroke={r.color ?? 'var(--accent-warning)'} strokeDasharray="3 3" />
          ))}
          <Line
            type="monotone"
            dataKey={signal}
            stroke={meta.color}
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 4, fill: meta.color }}
            isAnimationActive={false}
            name={meta.label}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
