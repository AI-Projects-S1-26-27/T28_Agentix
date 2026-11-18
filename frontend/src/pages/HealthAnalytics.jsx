import { useState, useEffect } from 'react'
import LiveChart from '../components/LiveChart'
import { useConnection } from '../context/ConnectionContext'
import { api } from '../services/api'
import store from '../services/dataStore'
import { fmt, fmtRelative, SIGNAL_LABELS } from '../utils/formatters'
import { HeartPulse, TrendingUp, TrendingDown, Minus } from 'lucide-react'

function StatTile({ label, value, unit, trend, color }) {
  const TrendIcon = trend === 'up' ? TrendingUp : trend === 'down' ? TrendingDown : Minus
  return (
    <div className="card rounded-xl p-4">
      <p className="text-xs mb-1" style={{ color: 'var(--text-muted)' }}>{label}</p>
      <div className="flex items-end gap-2">
        <span className="metric-value text-2xl font-bold" style={{ color: color ?? 'var(--text-primary)' }}>{value}</span>
        {unit && <span className="text-sm mb-0.5" style={{ color: 'var(--text-muted)' }}>{unit}</span>}
      </div>
      {trend && (
        <div className="flex items-center gap-1 mt-1">
          <TrendIcon size={12} style={{ color: trend === 'up' ? 'var(--accent-warning)' : trend === 'down' ? 'var(--accent-health)' : 'var(--text-muted)' }} />
          <span className="text-xs capitalize" style={{ color: 'var(--text-muted)' }}>{trend}</span>
        </div>
      )}
    </div>
  )
}

function getTrend(arr, key) {
  if (arr.length < 10) return 'stable'
  const half = Math.floor(arr.length / 2)
  const avg = vals => vals.reduce((s, v) => s + (v[key] ?? 0), 0) / vals.length
  const early = avg(arr.slice(0, half))
  const late  = avg(arr.slice(half))
  if (late > early * 1.03) return 'up'
  if (late < early * 0.97) return 'down'
  return 'stable'
}

export default function HealthAnalytics() {
  const { latestReading } = useConnection()
  const [latest, setLatest]   = useState(store.latest)
  const [history, setHistory] = useState(store.history)

  // Real-time updates from app-level WebSocket
  useEffect(() => {
    if (!latestReading) return
    setLatest(latestReading)
    setHistory([...store.history])
  }, [latestReading])

  // Initial load
  useEffect(() => {
    if (!store.latest) {
      api.readings.latest()
        .then(r => { if (r?.reading) { store.latest = r.reading; setLatest(r.reading) } })
        .catch(() => {})
    }
    if (store.history.length === 0) {
      api.history.readings({ limit: 120 })
        .then(r => { if (r?.length) { store.history = r.slice().reverse(); setHistory(store.history) } })
        .catch(() => {})
    }
  }, [])

  // Available signals from WESAD real dataset only
  const signals = ['heart_rate', 'body_temperature', 'eda']

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold" style={{ color: 'var(--text-primary)' }}>
          <HeartPulse size={22} className="inline mr-2 mb-0.5" style={{ color: 'var(--accent-danger)' }} />
          Health Analytics
        </h1>
        <p className="text-sm mt-0.5" style={{ color: 'var(--text-muted)' }}>Biometric signals from WESAD wearable dataset</p>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
        {signals.map(sig => {
          const meta = SIGNAL_LABELS[sig] ?? { label: sig, unit: '', color: 'var(--accent-primary)' }
          return (
            <StatTile
              key={sig}
              label={meta.label}
              value={fmt(latest?.[sig], sig === 'heart_rate' ? 0 : 1)}
              unit={meta.unit}
              trend={getTrend(history, sig)}
              color={meta.color}
            />
          )
        })}
      </div>

      <div>
        <h2 className="text-base font-semibold mb-3" style={{ color: 'var(--text-secondary)' }}>Trend Charts</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {(() => { const w = history.slice(-30); return (<>
          <LiveChart signal="heart_rate"       data={w} refLines={[{ value: 100 }, { value: 50 }]} />
          <LiveChart signal="body_temperature" data={w} refLines={[{ value: 38 }, { value: 35 }]} />
          <LiveChart signal="eda"              data={w} />
          </>) })()}
        </div>
      </div>

      {latest && (
        <div className="card rounded-xl p-5">
          <h3 className="text-sm font-semibold mb-3" style={{ color: 'var(--text-secondary)' }}>Current Reading</h3>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
            {signals.map(sig => {
              const meta = SIGNAL_LABELS[sig] ?? { label: sig, unit: '' }
              return (
                <div key={sig}>
                  <p className="text-xs" style={{ color: 'var(--text-muted)' }}>{meta.label}</p>
                  <p className="text-lg font-bold" style={{ color: meta.color ?? 'var(--text-primary)' }}>
                    {fmt(latest[sig])} <span className="text-xs font-normal" style={{ color: 'var(--text-muted)' }}>{meta.unit}</span>
                  </p>
                </div>
              )
            })}
          </div>
          <p className="text-xs mt-3" style={{ color: 'var(--text-muted)' }}>{fmtRelative(latest.timestamp)}</p>
        </div>
      )}
    </div>
  )
}
