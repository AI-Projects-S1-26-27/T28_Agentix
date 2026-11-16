import { useCallback } from 'react'
import { Routes, Route } from 'react-router-dom'
import { ThemeProvider } from './context/ThemeContext'
import { ConnectionProvider, useConnection } from './context/ConnectionContext'
import { useWebSocket } from './hooks/useWebSocket'
import DashboardLayout from './layouts/DashboardLayout'
import Dashboard from './pages/Dashboard'
import HealthAnalytics from './pages/HealthAnalytics'
import EnvironmentalIntelligence from './pages/EnvironmentalIntelligence'
import AIRiskAnalysis from './pages/AIRiskAnalysis'
import ScenarioSimulator from './pages/ScenarioSimulator'
import Alerts from './pages/Alerts'
import store from './services/dataStore'

/**
 * Inner component that sits inside ConnectionProvider so it can call
 * useConnection(). Owns the single app-level WebSocket connection.
 * Pages read latestReading from context instead of opening their own WS.
 */
function AppRoutes() {
  const { setLatestReading } = useConnection()

  const handleReading = useCallback((data) => {
    // Update shared store (persists across navigation)
    store.latest = data
    const entry = { ...data, timestamp: data.timestamp ?? Date.now() }
    store.history    = [...store.history,    entry].slice(-120)
    store.envHistory = [...store.envHistory, entry].slice(-120)
    // Notify all mounted pages via context
    setLatestReading(data)
  }, [setLatestReading])

  useWebSocket(handleReading)

  return (
    <DashboardLayout>
      <Routes>
        <Route path="/"            element={<Dashboard />} />
        <Route path="/health"      element={<HealthAnalytics />} />
        <Route path="/environment" element={<EnvironmentalIntelligence />} />
        <Route path="/risk"        element={<AIRiskAnalysis />} />
        <Route path="/simulator"   element={<ScenarioSimulator />} />
        <Route path="/alerts"      element={<Alerts />} />
      </Routes>
    </DashboardLayout>
  )
}

export default function App() {
  return (
    <ThemeProvider>
      <ConnectionProvider>
        <AppRoutes />
      </ConnectionProvider>
    </ThemeProvider>
  )
}
