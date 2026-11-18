import { useState, useEffect } from 'react'
import { useConnection } from '../context/ConnectionContext'
import HealthCard from '../components/HealthCard'
import EnvironmentCard from '../components/EnvironmentCard'
import RiskCard from '../components/RiskCard'
import LiveChart from '../components/LiveChart'
import { api } from '../services/api'
import store from '../services/dataStore'
import { Activity, Wifi, WifiOff } from 'lucide-react'

function aqiFromPm25(pm25) {
  if (pm25 == null) return null
  const c = Math.floor(pm25 * 10) / 10
  const bp = [[0,12,0,50],[12.1,35.4,51,100],[35.5,55.4,101,150],[55.5,150.4,151,200],[150.5,250.4,201,300],[250.5,500.4,301,500]]
  for (const [cLo,cHi,aLo,aHi] of bp) if (c>=cLo&&c<=cHi) return Math.round(((aHi-aLo)/(cHi-cLo))*(c-cLo)+aLo)
  return 500
}
function aqiMeta(aqi) {
  if (aqi==null) return { label:'—', color:'var(--text-muted)' }
  if (aqi<=50)  return { label:'Good',           color:'var(--accent-health)' }
  if (aqi<=100) return { label:'Moderate',        color:'var(--accent-warning)' }
  if (aqi<=150) return { label:'Sensitive Groups',color:'var(--accent-warning)' }
  if (aqi<=200) return { label:'Unhealthy',       color:'var(--accent-danger)' }
  return              { label:'Very Unhealthy',   color:'var(--accent-danger)' }
}

export default function Dashboard() {
  const { isConnected, latestReading } = useConnection()
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
          {(() => { const w = history.slice(-30); return (<>
          <LiveChart signal="heart_rate"          data={w} refLines={[{ value: 100, color: 'var(--accent-warning)' }, { value: 50, color: 'var(--accent-warning)' }]} />
          <LiveChart signal="body_temperature"    data={w} refLines={[{ value: 38, color: 'var(--accent-warning)' }]} />
          <LiveChart signal="eda"                 data={w} />
          <LiveChart signal="pm25"               data={w} refLines={[{ value: 35, color: 'var(--accent-warning)' }]} />
          <LiveChart signal="humidity"            data={w} />
          <LiveChart signal="ambient_temperature" data={w} />
          </>) })()}
        </div>
      </div>
    </div>
  )
}
