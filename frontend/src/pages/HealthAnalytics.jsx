import { useState, useEffect, useCallback } from 'react'
import LiveChart from '../components/LiveChart'
import { useWebSocket } from '../hooks/useWebSocket'
import { api } from '../services/api'
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
  const late = avg(arr.slice(half))
  if (late > early * 1.03) return 'up'
  if (late < early * 0.97) return 'down'
  return 'stable'
}

export default function HealthAnalytics() {
  const [latest, setLatest] = useState(null)
  const [history, setHistory] = useState([])

  const handleReading = useCallback((data) => {
    setLatest(data)
    setHistory(prev => [...prev, { ...data, timestamp: data.timestamp ?? Date.now() }].slice(-120))
  }, [])

  useWebSocket(handleReading)

  useEffect(() => {
    api.history.readings({ limit: 120 }).then(r => setHistory(r ?? [])).catch(() => {})
  }, [])

  const signals = ['heart_rate', 'spo2', 'body_temperature', 'activity_level', 'eda']

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold" style={{ color: 'var(--text-primary)' }}>
          <HeartPulse size={22} className="inline mr-2 mb-0.5" style={{ color: 'var(--accent-danger)' }} />
          Health Analytics
        </h1>
        <p className="text-sm mt-0.5" style={{ color: 'var(--text-muted)' }}>Biometric signals from wearable sensors</p>
      </div>

      {/* Quick stats */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
        {signals.map(sig => {
          const meta = SIGNAL_LABELS[sig] ?? { label: sig, unit: '', color: 'var(--accent-primary)' }
          return (
            <StatTile
              key={sig}
              label={meta.label}
              value={fmt(latest?.[sig], sig === 'heart_rate' || sig === 'activity_level' ? 0 : 1)}
              unit={meta.unit}
              trend={getTrend(history, sig)}
              color={meta.color}
            />
          )
        })}
      </div>

      {/* Trend charts */}
      <div>
        <h2 className="text-base font-semibold mb-3" style={{ color: 'var(--text-secondary)' }}>Trend Charts</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <LiveChart signal="heart_rate" data={history} refLines={[{ value: 100 }, { value: 50 }]} />
          <LiveChart signal="spo2" data={history} refLines={[{ value: 95, color: 'var(--accent-danger)' }]} />
          <LiveChart signal="body_temperature" data={history} refLines={[{ value: 38 }, { value: 35 }]} />
          <LiveChart signal="activity_level" data={history} />
          <LiveChart signal="eda" data={history} />
        </div>
      </div>

      {/* Baseline comparison */}
      {latest && (
        <div className="card rounded-xl p-5">
          <h3 className="text-sm font-semibold mb-3" style={{ color: 'var(--text-secondary)' }}>Current Reading</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
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
