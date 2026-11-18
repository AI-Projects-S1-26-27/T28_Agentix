import { useState, useEffect, useCallback } from 'react'
import AlertCard from '../components/AlertCard'
import { useConnection } from '../context/ConnectionContext'
import { api } from '../services/api'
import { BellRing, CheckCheck, Inbox, Filter } from 'lucide-react'

const FILTERS = ['all', 'info', 'warning', 'critical', 'unacknowledged']

export default function Alerts() {
  const { latestReading } = useConnection()
  const [alerts, setAlerts] = useState([])
  const [filter, setFilter] = useState('all')
  const [loading, setLoading] = useState(true)
  const [ackingAll, setAckingAll] = useState(false)

  const fetchAlerts = useCallback(() => {
    api.alerts.list().then(r => { setAlerts(r ?? []); setLoading(false) }).catch(() => setLoading(false))
  }, [])

  // Initial load + periodic refresh
  useEffect(() => {
    fetchAlerts()
    const interval = setInterval(fetchAlerts, 10000)
    return () => clearInterval(interval)
  }, [fetchAlerts])

  // Refresh on new reading from app-level WebSocket
  useEffect(() => {
    if (!latestReading) return
    fetchAlerts()
  }, [latestReading, fetchAlerts])

  async function acknowledge(id) {
    try {
      await api.alerts.acknowledge(id)
      setAlerts(prev => prev.map(a => a.id === id ? { ...a, acknowledged: true } : a))
    } catch (e) {
      console.error('Failed to acknowledge alert', e)
    }
  }

  async function acknowledgeAll() {
    setAckingAll(true)
    try {
      const unacked = alerts.filter(a => !a.acknowledged)
      await Promise.allSettled(unacked.map(a => api.alerts.acknowledge(a.id)))
      setAlerts(prev => prev.map(a => ({ ...a, acknowledged: true })))
    } finally {
      setAckingAll(false)
    }
  }

  const filtered = alerts.filter(a => {
    if (filter === 'all') return true
    if (filter === 'unacknowledged') return !a.acknowledged
    return a.level === filter
  })

  const unackedCount = alerts.filter(a => !a.acknowledged).length
  const counts = {
    all: alerts.length,
    info: alerts.filter(a => a.level === 'info').length,
    warning: alerts.filter(a => a.level === 'warning').length,
    critical: alerts.filter(a => a.level === 'critical').length,
    unacknowledged: unackedCount,
  }

  return (
    <div className="space-y-6">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold" style={{ color: 'var(--text-primary)' }}>
            <BellRing size={22} className="inline mr-2 mb-0.5" style={{ color: 'var(--accent-warning)' }} />
            Alerts
            {unackedCount > 0 && (
              <span className="ml-2 text-sm font-semibold px-2 py-0.5 rounded-full" style={{ background: 'var(--accent-danger)', color: 'white' }}>
                {unackedCount}
              </span>
            )}
          </h1>
          <p className="text-sm mt-0.5" style={{ color: 'var(--text-muted)' }}>Health and environment alert notifications</p>
        </div>

        {unackedCount > 0 && (
          <button
            onClick={acknowledgeAll}
            disabled={ackingAll}
            className="flex items-center gap-2 text-sm px-3 py-1.5 rounded-lg border transition-colors"
            style={{ borderColor: 'var(--border)', color: 'var(--text-secondary)' }}
            onMouseEnter={e => e.currentTarget.style.background = 'var(--surface-elevated)'}
            onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
          >
            <CheckCheck size={14} />
            Ack All ({unackedCount})
          </button>
        )}
      </div>

      {/* Filter tabs */}
      <div className="flex items-center gap-1 p-1 rounded-lg" style={{ background: 'var(--surface)' }}>
        <Filter size={12} className="ml-1" style={{ color: 'var(--text-muted)' }} />
        {FILTERS.map(f => (
          <button
            key={f}
            onClick={() => setFilter(f)}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-all capitalize"
            style={{
              background: filter === f ? 'var(--surface-elevated)' : 'transparent',
              color: filter === f ? 'var(--text-primary)' : 'var(--text-muted)',
              boxShadow: filter === f ? '0 1px 3px color-mix(in srgb, black 15%, transparent)' : 'none',
            }}
          >
            {f.replace('_', ' ')}
            {counts[f] > 0 && (
              <span className="text-[10px] px-1 py-0.5 rounded-full" style={{ background: filter === f ? 'var(--accent-primary)' : 'var(--border)', color: filter === f ? 'white' : 'var(--text-muted)' }}>
                {counts[f]}
              </span>
            )}
          </button>
        ))}
      </div>

      {/* Alert list */}
      {loading ? (
        <div className="space-y-2">
          {[1,2,3].map(i => <div key={i} className="skeleton h-16 rounded-lg" />)}
        </div>
      ) : filtered.length === 0 ? (
        <div className="card rounded-xl p-12 text-center">
          <Inbox size={40} className="mx-auto mb-3" style={{ color: 'var(--text-muted)', opacity: 0.3 }} />
          <p className="font-medium" style={{ color: 'var(--text-secondary)' }}>No alerts{filter !== 'all' ? ` for filter "${filter.replace('_', ' ')}"` : ''}</p>
          <p className="text-sm mt-1" style={{ color: 'var(--text-muted)' }}>Alerts appear here when thresholds are exceeded.</p>
        </div>
      ) : (
        <div className="space-y-2">
          {filtered.map((alert, i) => (
            <AlertCard key={alert.id ?? i} alert={alert} onAcknowledge={acknowledge} />
          ))}
        </div>
      )}
    </div>
  )
}
