import { useState } from 'react'
import Navbar from '../components/Navbar'
import Sidebar from '../components/Sidebar'

export default function DashboardLayout({ children }) {
  const [sidebarOpen, setSidebarOpen] = useState(false)

  return (
    <div className="min-h-screen" style={{ background: 'var(--bg)' }}>
      <Navbar
        sidebarOpen={sidebarOpen}
        onToggleSidebar={() => setSidebarOpen(o => !o)}
      />

      <Sidebar
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
      />

      {/* Main content — offset for fixed navbar + desktop sidebar */}
      <main
        className="pt-14 min-h-screen transition-all duration-200"
        style={{ marginLeft: 'clamp(0px, 220px, 220px)' }}
      >
        {/* Only apply desktop sidebar offset */}
        <div className="hidden lg:block" style={{ width: 0 }} />
        <div className="lg:pl-0 p-6 max-w-screen-xl mx-auto">
          {children}
        </div>
      </main>
    </div>
  )
}
