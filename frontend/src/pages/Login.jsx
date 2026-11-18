import { useState } from 'react'
import { Activity } from 'lucide-react'

export default function Login({ onLogin }) {
  const [email, setEmail]       = useState('')
  const [password, setPassword] = useState('')
  const [error, setError]       = useState('')
  const [loading, setLoading]   = useState(false)

  function handleSubmit(e) {
    e.preventDefault()
    if (!email || !password) { setError('Please fill in all fields.'); return }
    setError('')
    setLoading(true)
    setTimeout(() => { setLoading(false); onLogin() }, 900)
  }

  return (
    <div className="min-h-screen flex items-center justify-center px-4"
      style={{ background: 'var(--bg)' }}>
      <div className="w-full max-w-sm">

        {/* Logo */}
        <div className="flex flex-col items-center mb-8">
          <div className="w-12 h-12 rounded-2xl flex items-center justify-center mb-3"
            style={{ background: 'var(--accent-danger)' }}>
            <Activity size={24} color="#fff" />
          </div>
          <h1 className="text-2xl font-bold" style={{ color: 'var(--text-primary)' }}>HealthSensing</h1>
          <p className="text-sm mt-1" style={{ color: 'var(--text-muted)' }}>Sign in to your dashboard</p>
        </div>

        {/* Card */}
        <div className="card rounded-2xl p-6 space-y-4">
          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="block text-xs font-medium mb-1.5" style={{ color: 'var(--text-secondary)' }}>
                Email address
              </label>
              <input
                type="email"
                autoComplete="email"
                value={email}
                onChange={e => setEmail(e.target.value)}
                placeholder="you@example.com"
                className="w-full px-3 py-2.5 rounded-xl text-sm outline-none transition-all"
                style={{
                  background: 'var(--surface)',
                  border: '1px solid var(--border)',
                  color: 'var(--text-primary)',
                }}
                onFocus={e => e.target.style.borderColor = 'var(--accent-primary)'}
                onBlur={e  => e.target.style.borderColor = 'var(--border)'}
              />
            </div>

            <div>
              <div className="flex items-center justify-between mb-1.5">
                <label className="text-xs font-medium" style={{ color: 'var(--text-secondary)' }}>
                  Password
                </label>
                <span className="text-xs cursor-pointer" style={{ color: 'var(--accent-primary)' }}>
                  Forgot password?
                </span>
              </div>
              <input
                type="password"
                autoComplete="current-password"
                value={password}
                onChange={e => setPassword(e.target.value)}
                placeholder="••••••••"
                className="w-full px-3 py-2.5 rounded-xl text-sm outline-none transition-all"
                style={{
                  background: 'var(--surface)',
                  border: '1px solid var(--border)',
                  color: 'var(--text-primary)',
                }}
                onFocus={e => e.target.style.borderColor = 'var(--accent-primary)'}
                onBlur={e  => e.target.style.borderColor = 'var(--border)'}
              />
            </div>

            {error && (
              <p className="text-xs px-3 py-2 rounded-lg"
                style={{ background: 'color-mix(in srgb, var(--accent-danger) 12%, var(--surface))', color: 'var(--accent-danger)' }}>
                {error}
              </p>
            )}

            <button
              type="submit"
              disabled={loading}
              className="w-full py-2.5 rounded-xl text-sm font-semibold transition-opacity"
              style={{
                background: 'var(--accent-danger)',
                color: '#fff',
                opacity: loading ? 0.7 : 1,
                cursor: loading ? 'default' : 'pointer',
              }}>
              {loading ? 'Signing in…' : 'Sign in'}
            </button>
          </form>

          <p className="text-center text-xs" style={{ color: 'var(--text-muted)' }}>
            Don't have an account?{' '}
            <span className="cursor-pointer" style={{ color: 'var(--accent-primary)' }}>Request access</span>
          </p>
        </div>

        <p className="text-center text-xs mt-6" style={{ color: 'var(--text-muted)' }}>
          © 2024 HealthSensing · Real-time biometric monitoring
        </p>
      </div>
    </div>
  )
}
