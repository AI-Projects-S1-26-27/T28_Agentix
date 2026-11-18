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
import Alerts from './pages/Alerts'
import UserContext from './pages/UserContext'
import store from './services/dataStore'

function AppRoutes() {
  const { setLatestReading } = useConnection()

  const handleReading = useCallback((data) => {
    store.latest = data
    const entry = { ...data, timestamp: data.timestamp ?? Date.now() }
    store.history    = [...store.history,    entry].slice(-120)
    store.envHistory = [...store.envHistory, entry].slice(-120)
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
        <Route path="/alerts"      element={<Alerts />} />
        <Route path="/context"     element={<UserContext />} />
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
