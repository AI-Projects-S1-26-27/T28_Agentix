import { useState, useEffect } from 'react'
import { useConnection } from '../context/ConnectionContext'
import HealthCard from '../components/HealthCard'
import EnvironmentCard from '../components/EnvironmentCard'
import RiskCard from '../components/RiskCard'
import LiveChart from '../components/LiveChart'
import { api } from '../services/api'
import store from '../services/dataStore'
import { Activity, Wifi, WifiOff } from 'lucide-react'

export default function Dashboard() {
  const { isConnected, latestReading } = useConnection()
  // isConnected used only for status badge
  const [latest, setLatest] = useState(store.latest)
  const [riskData, setRiskData] = useState(store.riskData)
  const [history, setHistory] = useState(store.history)

  // Real-time updates from app-level WebSocket (via context)
  useEffect(() => {
    if (!latestReading) return
    setLatest(latestReading)
    setHistory([...store.history])
  }, [latestReading])

  // Refresh risk score whenever a new reading arrives (WS or polling)
  useEffect(() => {
    if (!latestReading) return
    api.risks.latest()
      .then(r => { store.riskData = r; setRiskData(r) })
      .catch(() => {})
  }, [latestReading])

  // Initial load on first mount (skipped if store already has data)
  useEffect(() => {
    if (!store.latest) {
      api.readings.latest()
        .then(r => { if (r?.reading) { store.latest = r.reading; setLatest(r.reading) } })
        .catch(() => {})
    }
    if (!store.riskData) {
      api.risks.latest()
        .then(r => { store.riskData = r; setRiskData(r) })
        .catch(() => {})
    }
    if (store.history.length === 0) {
      api.history.readings({ limit: 60 })
        .then(r => { if (r?.length) { store.history = r.slice().reverse(); setHistory(store.history) } })
        .catch(() => {})
    }
  }, [])

  // App-level polling (App.jsx) handles WS-down fallback for readings.
  // Per-page polling removed to avoid duplicate intervals and stale context.

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold" style={{ color: 'var(--text-primary)' }}>Dashboard</h1>
          <p className="text-sm mt-0.5" style={{ color: 'var(--text-muted)' }}>Real-time health &amp; environment overview</p>
        </div>
        <div className="flex items-center gap-2 px-3 py-1.5 rounded-full" style={{
          background: isConnected ? 'color-mix(in srgb, var(--accent-health) 12%, var(--surface))' : 'color-mix(in srgb, var(--accent-danger) 10%, var(--surface))',
          border: `1px solid ${isConnected ? 'color-mix(in srgb, var(--accent-health) 25%, transparent)' : 'color-mix(in srgb, var(--accent-danger) 25%, transparent)'}`
        }}>
          {isConnected
            ? <Wifi size={14} style={{ color: 'var(--accent-health)' }} />
            : <WifiOff size={14} style={{ color: 'var(--accent-danger)' }} />}
          <span className="text-xs font-medium" style={{ color: isConnected ? 'var(--accent-health)' : 'var(--accent-danger)' }}>
            {isConnected ? 'Connected' : 'Disconnected'}
          </span>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <HealthCard reading={latest} />
        <EnvironmentCard reading={latest} />
        <RiskCard risk={riskData} />
      </div>

      <div>
        <h2 className="text-base font-semibold mb-3" style={{ color: 'var(--text-secondary)' }}>
          <Activity size={16} className="inline mr-2" style={{ color: 'var(--accent-primary)' }} />
          Live Signals
        </h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          <LiveChart signal="heart_rate"          data={history} refLines={[{ value: 100, color: 'var(--accent-warning)' }, { value: 50, color: 'var(--accent-warning)' }]} />
          <LiveChart signal="spo2"                data={history} refLines={[{ value: 95, color: 'var(--accent-danger)' }]} />
          <LiveChart signal="body_temperature"    data={history} refLines={[{ value: 38, color: 'var(--accent-warning)' }]} />
          <LiveChart signal="aqi"                 data={history} refLines={[{ value: 100, color: 'var(--accent-warning)' }]} />
          <LiveChart signal="humidity"            data={history} />
          <LiveChart signal="ambient_temperature" data={history} />
        </div>
      </div>
    </div>
  )
}
