/**
 * contextStore — persists selected user contexts to localStorage
 * and syncs with the backend so risk thresholds adjust accordingly.
 *
 * Available context keys:
 *   exercising     — higher HR acceptable
 *   air_polluted   — stricter PM2.5 warnings
 *   resting        — lower HR thresholds
 *   high_stress    — lower EDA threshold
 */

const STORAGE_KEY = 'hs_context_flags'
const API_URL     = '/api/context'

function load() {
  try { return JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]') }
  catch { return [] }
}

function save(flags) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(flags))
  fetch(API_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ context_flags: flags }),
  }).catch(() => {})
}

const contextStore = {
  get flags() { return load() },

  toggle(key) {
    const current = load()
    const next = current.includes(key)
      ? current.filter(k => k !== key)
      : [...current, key]
    save(next)
    return next
  },

  set(flags) { save(flags) },

  isActive(key) { return load().includes(key) },

  /** Returns refLine configs for a given signal based on active contexts. */
  refLines(signal) {
    const flags = load()
    const exercising   = flags.includes('exercising')
    const airPolluted  = flags.includes('air_polluted')
    const resting      = flags.includes('resting')

    if (signal === 'heart_rate') {
      if (exercising) return [{ value: 170, color: 'var(--accent-warning)' }, { value: 45, color: 'var(--accent-warning)' }]
      if (resting)    return [{ value: 80,  color: 'var(--accent-warning)' }, { value: 55, color: 'var(--accent-warning)' }]
      return                 [{ value: 100, color: 'var(--accent-warning)' }, { value: 50, color: 'var(--accent-warning)' }]
    }
    if (signal === 'pm25') {
      if (airPolluted) return [{ value: 15, color: 'var(--accent-danger)' }, { value: 35, color: 'var(--accent-warning)' }]
      return                  [{ value: 35, color: 'var(--accent-warning)' }]
    }
    if (signal === 'body_temperature') {
      if (exercising) return [{ value: 39.0, color: 'var(--accent-warning)' }]
      return                 [{ value: 38.0, color: 'var(--accent-warning)' }]
    }
    if (signal === 'eda') {
      if (flags.includes('high_stress')) return [{ value: 5, color: 'var(--accent-warning)' }]
      return [{ value: 10, color: 'var(--accent-warning)' }]
    }
    if (signal === 'ambient_temperature') return [{ value: 35, color: 'var(--accent-warning)' }]
    if (signal === 'humidity')            return [{ value: 80, color: 'var(--accent-warning)' }]
    return []
  },
}

export default contextStore
