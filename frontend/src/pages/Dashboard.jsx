import { useState, useEffect, useCallback } from 'react'
import { useConnection } from '../context/ConnectionContext'
import HealthCard from '../components/HealthCard'
import EnvironmentCard from '../components/EnvironmentCard'
import RiskCard from '../components/RiskCard'
import LiveChart from '../components/LiveChart'
import { useWebSocket } from '../hooks/useWebSocket'
import { api } from '../services/api'
import { Activity, Wifi, WifiOff } from 'lucide-react'

export default function Dashboard() {
  const { isConnected } = useConnection()
  const [latest, setLatest] = useState(null)
  const [riskData, setRiskData] = useState(null)
  const [history, setHistory] = useState([])

  const handleReading = useCallback((data) => {
    setLatest(data)
    setHistory(prev => [...prev, { ...data, timestamp: data.timestamp ?? Date.now() }].slice(-60))
  }, [])

  useWebSocket(handleReading)

  useEffect(() => {
    api.risks.latest().then(setRiskData).catch(() => {})
    api.history.readings({ limit: 60 }).then(r => setHistory(r ?? [])).catch(() => {})
  }, [])

  useEffect(() => {
    if (!latest) return
    api.risks.latest().then(setRiskData).catch(() => {})
  }, [latest])

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold" style={{ color: 'var(--text-primary)' }}>Dashboard</h1>
          <p className="text-sm mt-0.5" style={{ color: 'var(--text-muted)' }}>Real-time health & environment overview</p>
        </div>
        <div className="flex items-center gap-2 px-3 py-1.5 rounded-full" style={{ background: isConnected ? 'color-mix(in srgb, var(--accent-health) 12%, var(--surface))' : 'color-mix(in srgb, var(--accent-danger) 10%, var(--surface))', border: `1px solid ${isConnected ? 'color-mix(in srgb, var(--accent-health) 25%, transparent)' : 'color-mix(in srgb, var(--accent-danger) 25%, transparent)'}` }}>
          {isConnected ? <Wifi size={14} style={{ color: 'var(--accent-health)' }} /> : <WifiOff size={14} style={{ color: 'var(--accent-danger)' }} />}
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
          <LiveChart signal="heart_rate" data={history} refLines={[{ value: 100, color: 'var(--accent-warning)' }, { value: 50, color: 'var(--accent-warning)' }]} />
          <LiveChart signal="spo2" data={history} refLines={[{ value: 95, color: 'var(--accent-danger)' }]} />
          <LiveChart signal="body_temperature" data={history} refLines={[{ value: 38, color: 'var(--accent-warning)' }]} />
          <LiveChart signal="aqi" data={history} refLines={[{ value: 100, color: 'var(--accent-warning)' }]} />
          <LiveChart signal="humidity" data={history} />
          <LiveChart signal="ambient_temperature" data={history} />
        </div>
      </div>
    </div>
  )
}
