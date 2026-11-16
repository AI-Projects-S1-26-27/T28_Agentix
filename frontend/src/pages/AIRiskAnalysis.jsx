import { useState, useEffect, useCallback } from 'react'
import RiskCard from '../components/RiskCard'
import useWebSocket from '../hooks/useWebSocket'
import { api } from '../services/api'
import { getRiskMeta, fmtRelative, fmtTime } from '../utils/formatters'
import { Brain, ChevronRight, Shield } from 'lucide-react'

function RiskHistoryRow({ item, onClick }) {
  const meta = getRiskMeta(item?.risk_level)
  return (
    <button
      onClick={onClick}
      className="w-full flex items-center gap-3 px-4 py-3 rounded-lg text-left transition-colors"
      style={{ background: 'var(--surface)' }}
      onMouseEnter={e => e.currentTarget.style.background = 'var(--surface-elevated)'}
      onMouseLeave={e => e.currentTarget.style.background = 'var(--surface)'}
    >
      <span className="text-xs font-bold px-2 py-1 rounded-md" style={{ background: meta.bg, color: meta.color }}>
        {meta.label}
      </span>
      <div className="flex-1 min-w-0">
        <p className="text-sm truncate" style={{ color: 'var(--text-secondary)' }}>
          {item?.explanation ?? 'Risk assessment recorded'}
        </p>
      </div>
      <span className="text-xs shrink-0" style={{ color: 'var(--text-muted)' }}>{fmtRelative(item?.timestamp)}</span>
      <ChevronRight size={14} style={{ color: 'var(--text-muted)' }} />
    </button>
  )
}

export default function AIRiskAnalysis() {
  const [latest, setLatest] = useState(null)
  const [history, setHistory] = useState([])
  const [selected, setSelected] = useState(null)
  const [loading, setLoading] = useState(true)

  const handleReading = useCallback(() => {
    api.risks.latest().then(r => { setLatest(r); setLoading(false) }).catch(() => setLoading(false))
  }, [])

  useWebSocket(handleReading)

  useEffect(() => {
    setLoading(true)
    Promise.all([
      api.risks.latest(),
      api.risks.history?.({ limit: 20 }) ?? Promise.resolve([]),
    ]).then(([l, h]) => {
      setLatest(l)
      setHistory(h ?? [])
      setLoading(false)
    }).catch(() => setLoading(false))
  }, [])

  const displayed = selected ?? latest

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold" style={{ color: 'var(--text-primary)' }}>
          <Brain size={22} className="inline mr-2 mb-0.5" style={{ color: 'var(--accent-primary)' }} />
          AI Risk Analysis
        </h1>
        <p className="text-sm mt-0.5" style={{ color: 'var(--text-muted)' }}>Composite health risk scoring from sensor fusion</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
        {/* Main risk panel */}
        <div className="lg:col-span-2 space-y-4">
          {loading ? (
            <div className="card rounded-xl p-6 space-y-3">
              {[1,2,3].map(i => <div key={i} className="skeleton h-5 rounded" />)}
            </div>
          ) : (
            <RiskCard risk={displayed} />
          )}

          {/* Risk level legend */}
          <div className="card rounded-xl p-4">
            <p className="text-xs font-semibold mb-3" style={{ color: 'var(--text-secondary)' }}>Risk Level Scale</p>
            {['low', 'moderate', 'high', 'very_high', 'severe'].map(level => {
              const m = getRiskMeta(level)
              return (
                <div key={level} className="flex items-center gap-2 py-1">
                  <Shield size={12} style={{ color: m.color }} />
                  <span className="text-xs" style={{ color: m.color }}>{m.label}</span>
                  <span className="text-xs" style={{ color: 'var(--text-muted)' }}>— severity {m.severity}</span>
                </div>
              )
            })}
          </div>
        </div>

        {/* History panel */}
        <div className="lg:col-span-3">
          <h2 className="text-base font-semibold mb-3" style={{ color: 'var(--text-secondary)' }}>Assessment History</h2>
          {history.length === 0 ? (
            <div className="card rounded-xl p-8 text-center">
              <Brain size={32} className="mx-auto mb-3" style={{ color: 'var(--text-muted)', opacity: 0.4 }} />
              <p className="text-sm" style={{ color: 'var(--text-muted)' }}>No history yet. Assessments appear here as data streams in.</p>
            </div>
          ) : (
            <div className="space-y-1.5">
              {history.map((item, i) => (
                <RiskHistoryRow key={i} item={item} onClick={() => setSelected(item)} />
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
