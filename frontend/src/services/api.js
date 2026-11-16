const BASE = '/api'

async function request(method, path, body) {
  const opts = { method, headers: { 'Content-Type': 'application/json' } }
  if (body) opts.body = JSON.stringify(body)
  const res = await fetch(BASE + path, opts)
  if (!res.ok) throw new Error(\API \ \ failed: \ \)
  return res.json()
}

export const api = {
  health: () => fetch('/health').then(r => r.json()),
  readings: {
    latest: () => request('GET', '/readings/latest'),
    list: (limit = 50) => request('GET', \/readings?limit=\),
    ingest: (m) => request('POST', '/readings', m),
  },
  risks: {
    latest: () => request('GET', '/risks/latest'),
    recent: (limit = 50) => request('GET', \/risks/recent?limit=\),
    evaluate: (body) => request('POST', '/risks/evaluate', body),
  },
  alerts: {
    active: () => request('GET', '/alerts/active'),
    list: (limit = 50, unacknowledgedOnly = false) =>
      request('GET', \/alerts?limit=\&unacknowledged_only=\),
    acknowledge: (id) => request('POST', \/alerts/\/acknowledge\),
  },
  history: {
    readings: ({ start, end, limit = 100, offset = 0 } = {}) => {
      const params = new URLSearchParams({ limit, offset })
      if (start) params.append('start', start)
      if (end) params.append('end', end)
      return request('GET', \/history/readings?\)
    },
    risks: (limit = 50) => request('GET', \/history/risks?limit=\),
    stats: () => request('GET', '/history/stats'),
  },
  scenarios: {
    list: () => request('GET', '/scenarios'),
    run: (scenarioId, count = 10) => request('POST', '/scenarios/run', { scenario_id: scenarioId, count }),
  },
  settings: {
    get: () => request('GET', '/settings'),
    update: (body) => request('PATCH', '/settings', body),
  },
  baseline: {
    get: (participantId = 'default') => request('GET', \/baseline?participant_id=\),
    calibrate: (body) => request('POST', '/baseline/calibrate', body),
  },
}

export default api
