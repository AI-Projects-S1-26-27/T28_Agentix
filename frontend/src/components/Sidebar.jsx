import { NavLink } from 'react-router-dom'
import {
  LayoutDashboard,
  HeartPulse,
  Wind,
  Brain,
  BellRing,
  SlidersHorizontal,
  X,
} from 'lucide-react'

const NAV_ITEMS = [
  { to: '/',            label: 'Dashboard',      Icon: LayoutDashboard },
  { to: '/health',      label: 'Health',         Icon: HeartPulse },
  { to: '/environment', label: 'Environment',    Icon: Wind },
  { to: '/risk',        label: 'Risk Analysis',  Icon: Brain },
  { to: '/alerts',      label: 'Alerts',         Icon: BellRing },
  { to: '/context',     label: 'My Context',     Icon: SlidersHorizontal },
]

export default function Sidebar({ open, onClose }) {
  return (
    <>
      {/* Mobile backdrop */}
      {open && (
        <div
          className="fixed inset-0 z-40 lg:hidden"
          style={{ background: 'rgba(0,0,0,0.5)' }}
          onClick={onClose}
        />
      )}

      {/* Sidebar panel */}
      <aside
        className={`
          fixed top-0 left-0 z-50 h-full flex flex-col pt-14
          transition-transform duration-200 ease-in-out
          lg:translate-x-0 lg:z-20
          ${open ? 'translate-x-0' : '-translate-x-full'}
        `}
        style={{
          width: '220px',
          background: 'var(--surface)',
          borderRight: '1px solid var(--border)',
        }}
      >
        {/* Mobile close */}
        <div className="flex items-center justify-between px-4 py-3 lg:hidden border-b" style={{ borderColor: 'var(--border)' }}>
          <span className="text-xs font-semibold uppercase tracking-widest" style={{ color: 'var(--text-muted)' }}>
            Navigation
          </span>
          <button onClick={onClose} className="p-1 rounded" style={{ color: 'var(--text-muted)' }}>
            <X size={16} />
          </button>
        </div>

        {/* Nav items */}
        <nav className="flex-1 overflow-y-auto px-3 py-4 space-y-0.5">
          {NAV_ITEMS.map(({ to, label, Icon }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              onClick={() => onClose?.()}
              className={({ isActive }) =>
                `nav-item ${isActive ? 'active' : ''}`
              }
            >
              <Icon size={17} strokeWidth={1.75} />
              <span>{label}</span>
            </NavLink>
          ))}
        </nav>

      </aside>
    </>
  )
}
