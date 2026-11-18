import { useState } from 'react'
import contextStore from '../services/contextStore'
import { Activity, Wind, Moon, Zap } from 'lucide-react'

const CONTEXTS = [
  {
    key: 'exercising',
    label: 'Exercising',
    description: 'Heart rate up to 170 bpm is acceptable. Temperature threshold raised.',
    Icon: Activity,
    color: 'var(--accent-health)',
  },
  {
    key: 'air_polluted',
    label: 'Air Polluted Area',
    description: 'Stricter PM2.5 warnings at 15 µg/m³ (WHO guideline) instead of 35.',
    Icon: Wind,
    color: 'var(--accent-env)',
  },
  {
    key: 'resting',
    label: 'Resting / Sleeping',
    description: 'Lower heart rate thresholds. Flags elevated HR sooner.',
    Icon: Moon,
    color: 'var(--accent-primary)',
  },
  {
    key: 'high_stress',
    label: 'High Stress',
    description: 'EDA warning triggered at lower skin conductance (5 µS).',
    Icon: Zap,
    color: 'var(--accent-warning)',
  },
]

export default function UserContext() {
  const [active, setActive] = useState(contextStore.flags)

  function toggle(key) {
    const next = contextStore.toggle(key)
    setActive([...next])
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold" style={{ color: 'var(--text-primary)' }}>My Context</h1>
        <p className="text-sm mt-0.5" style={{ color: 'var(--text-muted)' }}>
          Tell the system about your current situation. Risk thresholds and graph warning lines adjust accordingly.
        </p>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        {CONTEXTS.map(({ key, label, description, Icon, color }) => {
          const on = active.includes(key)
          return (
            <button
              key={key}
              onClick={() => toggle(key)}
              className="card rounded-xl p-5 text-left transition-all"
              style={{
                borderColor: on ? color : 'var(--border)',
                borderWidth: on ? 2 : 1,
                borderStyle: 'solid',
                background: on
                  ? `color-mix(in srgb, ${color} 8%, var(--surface))`
                  : 'var(--surface)',
                cursor: 'pointer',
              }}
            >
              <div className="flex items-center gap-3 mb-2">
                <div className="p-2 rounded-lg" style={{
                  background: `color-mix(in srgb, ${color} 15%, transparent)`,
                }}>
                  <Icon size={18} style={{ color }} />
                </div>
                <span className="font-semibold text-sm" style={{ color: on ? color : 'var(--text-primary)' }}>
                  {label}
                </span>
                {on && (
                  <span className="ml-auto text-xs font-medium px-2 py-0.5 rounded-full"
                    style={{ background: `color-mix(in srgb, ${color} 20%, transparent)`, color }}>
                    Active
                  </span>
                )}
              </div>
              <p className="text-xs" style={{ color: 'var(--text-muted)' }}>{description}</p>
            </button>
          )
        })}
      </div>

      {active.length > 0 && (
        <div className="card rounded-xl p-4" style={{ borderColor: 'var(--border)' }}>
          <p className="text-xs font-semibold mb-2" style={{ color: 'var(--text-muted)' }}>ACTIVE ADJUSTMENTS</p>
          <ul className="space-y-1 text-sm" style={{ color: 'var(--text-secondary)' }}>
            {active.includes('exercising')   && <li>• Heart rate warning raised to 170 bpm</li>}
            {active.includes('air_polluted') && <li>• PM2.5 warning lowered to 15 µg/m³</li>}
            {active.includes('resting')      && <li>• Heart rate warning lowered to 80 bpm</li>}
            {active.includes('high_stress')  && <li>• EDA (skin conductance) warning at 5 µS</li>}
          </ul>
        </div>
      )}
    </div>
  )
}
