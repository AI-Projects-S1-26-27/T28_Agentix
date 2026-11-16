import { Sun, Moon, Wifi, WifiOff, Activity } from 'lucide-react'
import { useTheme } from '../context/ThemeContext'
import { useConnection } from '../context/ConnectionContext'
import { fmtRelative } from '../utils/formatters'

export default function Navbar({ sidebarOpen, onToggleSidebar }) {
  const { theme, toggleTheme } = useTheme()
  const { isConnected, lastHeartbeat } = useConnection()

  return (
    <header
      className="fixed top-0 left-0 right-0 z-30 flex items-center justify-between px-4 h-14 border-b"
      style={{ background: 'var(--surface)', borderColor: 'var(--border)' }}
    >
      {/* Left — logo + hamburger */}
      <div className="flex items-center gap-3">
        <button
          onClick={onToggleSidebar}
          className="p-1.5 rounded-md transition-colors hover:bg-[var(--surface-elevated)] lg:hidden"
          aria-label="Toggle sidebar"
        >
          <span className="block w-5 h-0.5 bg-current mb-1" />
          <span className="block w-5 h-0.5 bg-current mb-1" />
          <span className="block w-4 h-0.5 bg-current" />
        </button>

        <div className="flex items-center gap-2">
          <div
            className="w-7 h-7 rounded-lg flex items-center justify-center"
            style={{ background: 'var(--accent-health)' }}
          >
            <Activity size={15} color="#fff" strokeWidth={2.5} />
          </div>
          <span className="text-sm font-semibold tracking-tight" style={{ color: 'var(--text-primary)' }}>
            HealthSensing
          </span>
        </div>
      </div>

      {/* Right — connection + theme */}
      <div className="flex items-center gap-2">
        {/* Connection status */}
        <div
          className="hidden sm:flex items-center gap-2 px-3 py-1.5 rounded-md text-xs font-medium border"
          style={{
            borderColor: isConnected ? 'color-mix(in srgb, var(--accent-health) 30%, var(--border))' : 'var(--border)',
            background: isConnected ? 'color-mix(in srgb, var(--accent-health) 8%, var(--surface-elevated))' : 'var(--surface-elevated)',
            color: isConnected ? 'var(--accent-health)' : 'var(--text-muted)',
          }}
        >
          <span
            className="w-1.5 h-1.5 rounded-full"
            style={{
              background: isConnected ? 'var(--accent-health)' : 'var(--text-muted)',
              boxShadow: isConnected ? '0 0 6px var(--accent-health)' : 'none',
            }}
          />
          {isConnected ? (
            <span>Live · {fmtRelative(lastHeartbeat)}</span>
          ) : (
            <span>Reconnecting…</span>
          )}
        </div>

        {/* Mobile connection dot */}
        <span
          className="sm:hidden w-2 h-2 rounded-full"
          style={{
            background: isConnected ? 'var(--accent-health)' : 'var(--text-muted)',
            boxShadow: isConnected ? '0 0 6px var(--accent-health)' : 'none',
          }}
        />

        {/* Theme toggle */}
        <button
          onClick={toggleTheme}
          className="p-2 rounded-md transition-colors"
          style={{ color: 'var(--text-secondary)' }}
          onMouseEnter={e => e.currentTarget.style.background = 'var(--surface-elevated)'}
          onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
          aria-label="Toggle theme"
        >
          {theme === 'dark' ? <Sun size={16} /> : <Moon size={16} />}
        </button>
      </div>
    </header>
  )
}
