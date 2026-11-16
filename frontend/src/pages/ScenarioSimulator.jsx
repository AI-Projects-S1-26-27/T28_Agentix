import { useState, useEffect } from 'react'
import { api } from '../services/api'
import { FlaskConical, Play, Square, CheckCircle2, AlertTriangle, Loader2 } from 'lucide-react'

const SCENARIOS = [
  {
    id: 'normal',
    label: 'Normal',
    description: 'Healthy resting state. All vitals within safe baseline ranges.',
    color: 'var(--accent-health)',
    icon: '🧘',
  },
  {
    id: 'heatwave',
    label: 'Heatwave',
    description: 'Elevated ambient temperature, rising body temp and heart rate. Dehydration risk.',
    color: 'var(--accent-warning)',
    icon: '🌡️',
  },
  {
    id: 'pollution',
    label: 'Air Pollution',
    description: 'High AQI and PM2.5 exposure. SpO₂ may drop. Respiratory stress indicators.',
    color: 'var(--accent-danger)',
    icon: '🏭',
  },
  {
    id: 'fatigue',
    label: 'Fatigue',
    description: 'Physical exertion and exhaustion. Elevated heart rate, reduced activity coordination.',
    color: 'var(--accent-critical)',
    icon: '😴',
  },
]

export default function ScenarioSimulator() {
  const [active, setActive] = useState(null)
  const [loading, setLoading] = useState(false)
  const [status, setStatus] = useState(null) // { type: 'success' | 'error', msg: string }

  useEffect(() => {
    api.scenarios.active?.().then(r => setActive(r?.scenario ?? null)).catch(() => {})
  }, [])

  async function activateScenario(id) {
    setLoading(true)
    setStatus(null)
    try {
      await api.scenarios.set(id)
      setActive(id)
      setStatus({ type: 'success', msg: `Scenario "${SCENARIOS.find(s => s.id === id)?.label}" activated.` })
    } catch (e) {
      setStatus({ type: 'error', msg: 'Failed to activate scenario. Is the backend running?' })
    } finally {
      setLoading(false)
    }
  }

  async function stopScenario() {
    setLoading(true)
    setStatus(null)
    try {
      await api.scenarios.set('normal')
      setActive('normal')
      setStatus({ type: 'success', msg: 'Scenario reset to normal.' })
    } catch (e) {
      setStatus({ type: 'error', msg: 'Failed to stop scenario.' })
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold" style={{ color: 'var(--text-primary)' }}>
          <FlaskConical size={22} className="inline mr-2 mb-0.5" style={{ color: 'var(--accent-warning)' }} />
          Scenario Simulator
        </h1>
        <p className="text-sm mt-0.5" style={{ color: 'var(--text-muted)' }}>Inject environmental and physiological stress scenarios for testing</p>
      </div>

      {/* Status feedback */}
      {status && (
        <div
          className="flex items-center gap-3 p-3 rounded-lg border"
          style={{
            background: status.type === 'success' ? 'color-mix(in srgb, var(--accent-health) 10%, var(--surface))' : 'color-mix(in srgb, var(--accent-danger) 10%, var(--surface))',
            borderColor: status.type === 'success' ? 'color-mix(in srgb, var(--accent-health) 25%, transparent)' : 'color-mix(in srgb, var(--accent-danger) 25%, transparent)',
          }}
        >
          {status.type === 'success'
            ? <CheckCircle2 size={16} style={{ color: 'var(--accent-health)' }} />
            : <AlertTriangle size={16} style={{ color: 'var(--accent-danger)' }} />
          }
          <p className="text-sm" style={{ color: 'var(--text-secondary)' }}>{status.msg}</p>
        </div>
      )}

      {/* Active indicator */}
      {active && active !== 'normal' && (
        <div className="flex items-center justify-between card rounded-xl p-4" style={{ borderColor: 'color-mix(in srgb, var(--accent-warning) 30%, var(--border))' }}>
          <div>
            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Active Scenario</p>
            <p className="font-semibold" style={{ color: 'var(--accent-warning)' }}>
              {SCENARIOS.find(s => s.id === active)?.icon} {SCENARIOS.find(s => s.id === active)?.label}
            </p>
          </div>
          <button
            onClick={stopScenario}
            disabled={loading}
            className="flex items-center gap-2 btn-ghost rounded-lg px-3 py-2 text-sm"
            style={{ color: 'var(--accent-danger)', borderColor: 'color-mix(in srgb, var(--accent-danger) 30%, transparent)' }}
          >
            {loading ? <Loader2 size={14} className="animate-spin" /> : <Square size={14} />}
            Stop
          </button>
        </div>
      )}

      {/* Scenario cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {SCENARIOS.map(sc => {
          const isActive = active === sc.id
          return (
            <div
              key={sc.id}
              className="card rounded-xl p-5 transition-all"
              style={{
                borderColor: isActive ? `color-mix(in srgb, ${sc.color} 40%, var(--border))` : 'var(--border)',
                background: isActive ? `color-mix(in srgb, ${sc.color} 6%, var(--surface))` : 'var(--surface)',
              }}
            >
              <div className="flex items-start justify-between gap-3 mb-3">
                <div>
                  <span className="text-2xl">{sc.icon}</span>
                  <p className="font-semibold mt-1" style={{ color: sc.color }}>{sc.label}</p>
                </div>
                {isActive && (
                  <span className="badge text-xs" style={{ background: `color-mix(in srgb, ${sc.color} 15%, transparent)`, color: sc.color }}>
                    Active
                  </span>
                )}
              </div>

              <p className="text-sm mb-4" style={{ color: 'var(--text-secondary)' }}>{sc.description}</p>

              <button
                onClick={() => isActive ? stopScenario() : activateScenario(sc.id)}
                disabled={loading}
                className="w-full flex items-center justify-center gap-2 py-2 px-4 rounded-lg text-sm font-medium transition-all"
                style={{
                  background: isActive ? `color-mix(in srgb, ${sc.color} 15%, transparent)` : `color-mix(in srgb, ${sc.color} 12%, transparent)`,
                  color: sc.color,
                  border: `1px solid color-mix(in srgb, ${sc.color} 30%, transparent)`,
                }}
              >
                {loading ? (
                  <Loader2 size={14} className="animate-spin" />
                ) : isActive ? (
                  <><Square size={14} /> Stop</>
                ) : (
                  <><Play size={14} /> Activate</>
                )}
              </button>
            </div>
          )
        })}
      </div>

      <div className="card rounded-xl p-4 text-sm" style={{ color: 'var(--text-muted)', borderColor: 'var(--border)' }}>
        <p className="font-medium mb-1" style={{ color: 'var(--text-secondary)' }}>How scenarios work</p>
        <p>Scenarios bias the simulator's data generation. The backend injects realistic physiological and environmental signals corresponding to each condition. Changes appear immediately on the Dashboard and trigger risk assessments within a few seconds.</p>
      </div>
    </div>
  )
}
