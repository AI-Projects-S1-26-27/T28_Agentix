import { useState, useEffect } from 'react'
import LiveChart from '../components/LiveChart'
import { useConnection } from '../context/ConnectionContext'
import { api } from '../services/api'
import store from '../services/dataStore'
import { fmt } from '../utils/formatters'
import { Wind, Thermometer, Droplets, AlertTriangle } from 'lucide-react'

function aqiFromPm25(pm25) {
  if (pm25 == null) return null
  const c = Math.floor(pm25 * 10) / 10
  const bp = [[0,12,0,50],[12.1,35.4,51,100],[35.5,55.4,101,150],[55.5,150.4,151,200],[150.5,250.4,201,300],[250.5,500.4,301,500]]
  for (const [cLo,cHi,aLo,aHi] of bp) if (c>=cLo&&c<=cHi) return Math.round(((aHi-aLo)/(cHi-cLo))*(c-cLo)+aLo)
  return 500
}
function aqiMeta(aqi) {
  if (aqi==null) return { label:'—', color:'var(--text-muted)' }
  if (aqi<=50)  return { label:'Good',          color:'var(--accent-health)' }
  if (aqi<=100) return { label:'Moderate',       color:'var(--accent-warning)' }
  if (aqi<=150) return { label:'Sensitive Groups',color:'var(--accent-warning)' }
  if (aqi<=200) return { label:'Unhealthy',      color:'var(--accent-danger)' }
  return              { label:'Very Unhealthy',  color:'var(--accent-danger)' }
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
  const { latestReading } = useConnection()
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
        .then(r => { if (r?.length) { store.envHistory = r.slice().reverse(); setHistory(store.envHistory) } })
        .catch(() => {})
    }
  }, [])

  const pm25 = latest?.pm25 ?? null
  const aqi  = aqiFromPm25(pm25)
  const aqiInfo = aqiMeta(aqi)

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold" style={{ color: 'var(--text-primary)' }}>
          <Wind size={22} className="inline mr-2 mb-0.5" style={{ color: 'var(--accent-env)' }} />
          Environmental Intelligence
        </h1>
        <p className="text-sm mt-0.5" style={{ color: 'var(--text-muted)' }}>Ambient conditions from Ljubljana air quality dataset</p>
      </div>

      {/* PM2.5 hero */}
      <div className="card-elevated rounded-xl p-5" style={{
        borderLeft: `4px solid ${(pm25 ?? 0) > 35 ? 'var(--accent-danger)' : (pm25 ?? 0) > 12 ? 'var(--accent-warning)' : 'var(--accent-health)'}`
      }}>
        <div className="flex items-start justify-between gap-4">
          <div>
            <p className="text-xs font-medium mb-1" style={{ color: 'var(--text-muted)' }}>Fine Particulate Matter (PM2.5)</p>
            <div className="flex items-baseline gap-3">
              <span className="text-5xl font-bold" style={{ color: (pm25 ?? 0) > 35 ? 'var(--accent-danger)' : (pm25 ?? 0) > 12 ? 'var(--accent-warning)' : 'var(--accent-health)' }}>
                {pm25 != null ? fmt(pm25, 1) : '—'}
              </span>
              <span className="text-sm" style={{ color: 'var(--text-muted)' }}>µg/m³</span>
            </div>
          </div>
          <div className="text-right">
            <p className="text-xs mb-1" style={{ color: 'var(--text-muted)' }}>AQI</p>
            <p className="text-3xl font-bold" style={{ color: aqiInfo.color }}>{aqi ?? '—'}</p>
            <p className="text-xs mt-0.5" style={{ color: aqiInfo.color }}>{aqiInfo.label}</p>
          </div>
        </div>
        {pm25 != null && pm25 > 35 && (
          <p className="text-xs mt-3 p-2 rounded-lg" style={{ background: 'color-mix(in srgb, var(--accent-danger) 10%, var(--surface))', color: 'var(--text-secondary)' }}>
            ⚠ PM2.5 exceeds safe threshold. Consider limiting outdoor exposure.
          </p>
        )}
      </div>

      <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
        <EnvStat icon={Thermometer} label="Ambient Temp"
          value={latest?.ambient_temperature != null ? fmt(latest.ambient_temperature) : null}
          unit="°C" warning={(latest?.ambient_temperature ?? 0) > 35} color="var(--accent-warning)" />
        <EnvStat icon={Droplets} label="Humidity"
          value={latest?.humidity != null ? fmt(latest.humidity, 0) : null}
          unit="%" warning={(latest?.humidity ?? 0) > 80} color="var(--accent-env)" />
        <EnvStat icon={AlertTriangle} label="PM2.5"
          value={pm25 != null ? fmt(pm25, 1) : null}
          unit="µg/m³" warning={(pm25 ?? 0) > 35} color="var(--accent-danger)" />
      </div>

      <div>
        <h2 className="text-base font-semibold mb-3" style={{ color: 'var(--text-secondary)' }}>Trend Charts</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {(() => { const w = history.slice(-30); return (<>
          <LiveChart signal="pm25"               data={w} refLines={[{ value: 35, color: 'var(--accent-warning)' }, { value: 15, color: 'var(--accent-health)' }]} />
          <LiveChart signal="ambient_temperature" data={w} refLines={[{ value: 35, color: 'var(--accent-warning)' }]} />
          <LiveChart signal="humidity"            data={w} refLines={[{ value: 80, color: 'var(--accent-warning)' }]} />
          </>) })()}
        </div>
      </div>
    </div>
  )
}
