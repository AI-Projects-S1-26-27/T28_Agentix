import { useState, useEffect } from 'react'
import LiveChart from '../components/LiveChart'
import { useConnection } from '../context/ConnectionContext'
import { api } from '../services/api'
import store from '../services/dataStore'
import { fmt, aqiCategory } from '../utils/formatters'
import { Wind, Thermometer, Droplets, AlertTriangle } from 'lucide-react'

function AqiBadge({ aqi }) {
  const cat = aqiCategory(aqi)
  return (
    <span className="badge text-xs font-semibold px-2 py-1" style={{ background: `color-mix(in srgb, ${cat.color} 15%, transparent)`, color: cat.color }}>
      {cat.label}
    </span>
  )
}

function EnvStat({ icon: Icon, label, value, unit, warning, color }) {
  return (
    <div className="card rounded-xl p-4" style={{ borderColor: warning ? `color-mix(in srgb, var(--accent-warning) 30%, var(--border))` : 'var(--border)' }}>
      <div className="flex items-center gap-2 mb-2">
        <Icon size={15} style={{ color: color ?? 'var(--accent-env)' }} />
        <span className="text-xs" style={{ color: 'var(--text-muted)' }}>{label}</span>
        {warning && <AlertTriangle size={12} style={{ color: 'var(--accent-warning)' }} className="ml-auto" />}
      </div>
      <div className="flex items-end gap-1">
        <span className="text-2xl font-bold" style={{ color: color ?? 'var(--text-primary)' }}>{value ?? '—'}</span>
        <span className="text-sm mb-0.5" style={{ color: 'var(--text-muted)' }}>{unit}</span>
      </div>
    </div>
  )
}

export default function EnvironmentalIntelligence() {
  const { isConnected, latestReading } = useConnection()
  const [latest, setLatest]   = useState(store.latest)
  const [history, setHistory] = useState(store.envHistory)

  // Real-time updates from app-level WebSocket
  useEffect(() => {
    if (!latestReading) return
    setLatest(latestReading)
    setHistory([...store.envHistory])
  }, [latestReading])

  // Initial load
  useEffect(() => {
    if (!store.latest) {
      api.readings.latest()
        .then(r => { if (r?.reading) { store.latest = r.reading; setLatest(r.reading) } })
        .catch(() => {})
    }
    if (store.envHistory.length === 0) {
      api.history.readings({ limit: 120 })
        .then(r => { if (r?.length) { store.envHistory = r; setHistory(r) } })
        .catch(() => {})
    }
  }, [])

  // Polling fallback when disconnected
  useEffect(() => {
    if (isConnected) return
    const latestId = setInterval(() => {
      api.readings.latest()
        .then(r => { if (r?.reading) { store.latest = r.reading; setLatest(r.reading) } })
        .catch(() => {})
    }, 2000)
    const histId = setInterval(() => {
      api.history.readings({ limit: 120 })
        .then(r => { if (r?.length) { store.envHistory = r; setHistory(r) } })
        .catch(() => {})
    }, 3000)
    return () => { clearInterval(latestId); clearInterval(histId) }
  }, [isConnected])

  const aqi    = latest?.aqi ?? null
  const aqiCat = aqiCategory(aqi ?? 0)

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold" style={{ color: 'var(--text-primary)' }}>
          <Wind size={22} className="inline mr-2 mb-0.5" style={{ color: 'var(--accent-env)' }} />
          Environmental Intelligence
        </h1>
        <p className="text-sm mt-0.5" style={{ color: 'var(--text-muted)' }}>Ambient conditions and air quality monitoring</p>
      </div>

      <div className="card-elevated rounded-xl p-5" style={{ borderLeft: `4px solid ${aqiCat.color}` }}>
        <div className="flex items-start justify-between gap-4">
          <div>
            <p className="text-xs font-medium mb-1" style={{ color: 'var(--text-muted)' }}>Air Quality Index</p>
            <div className="flex items-baseline gap-3">
              <span className="text-5xl font-bold" style={{ color: aqiCat.color }}>{aqi != null ? fmt(aqi, 0) : '—'}</span>
              {aqi != null && <AqiBadge aqi={aqi} />}
            </div>
          </div>
          <div className="text-right">
            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>PM2.5</p>
            <p className="text-xl font-bold" style={{ color: (latest?.pm25 ?? 0) > 35 ? 'var(--accent-danger)' : 'var(--text-primary)' }}>
              {latest?.pm25 != null ? fmt(latest.pm25) : '—'} <span className="text-xs font-normal" style={{ color: 'var(--text-muted)' }}>µg/m³</span>
            </p>
          </div>
        </div>
        {aqi != null && aqi > 100 && (
          <p className="text-xs mt-3 p-2 rounded-lg" style={{ background: `color-mix(in srgb, ${aqiCat.color} 10%, var(--surface))`, color: 'var(--text-secondary)' }}>
            ⚠ Air quality is {aqiCat.label.toLowerCase()}. Limit outdoor exposure.
          </p>
        )}
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <EnvStat icon={Thermometer} label="Ambient Temp"
          value={latest?.ambient_temperature != null ? fmt(latest.ambient_temperature) : null}
          unit="°C" warning={(latest?.ambient_temperature ?? 0) > 35} color="var(--accent-warning)" />
        <EnvStat icon={Droplets} label="Humidity"
          value={latest?.humidity != null ? fmt(latest.humidity, 0) : null}
          unit="%" warning={(latest?.humidity ?? 0) > 80} color="var(--accent-env)" />
        <EnvStat icon={Wind} label="AQI"
          value={aqi != null ? fmt(aqi, 0) : null}
          unit="" warning={(aqi ?? 0) > 100} color={aqiCat.color} />
        <EnvStat icon={AlertTriangle} label="PM2.5"
          value={latest?.pm25 != null ? fmt(latest.pm25) : null}
          unit="µg/m³" warning={(latest?.pm25 ?? 0) > 35} color="var(--accent-danger)" />
      </div>

      <div>
        <h2 className="text-base font-semibold mb-3" style={{ color: 'var(--text-secondary)' }}>Trend Charts</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <LiveChart signal="aqi"                 data={history} refLines={[{ value: 50, color: 'var(--accent-warning)' }, { value: 100, color: 'var(--accent-danger)' }]} />
          <LiveChart signal="pm25"               data={history} refLines={[{ value: 35, color: 'var(--accent-warning)' }]} />
          <LiveChart signal="ambient_temperature" data={history} refLines={[{ value: 35, color: 'var(--accent-warning)' }]} />
          <LiveChart signal="humidity"            data={history} refLines={[{ value: 80, color: 'var(--accent-warning)' }]} />
        </div>
      </div>
    </div>
  )
}
