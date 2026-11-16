import { Routes, Route } from 'react-router-dom'
import { ThemeProvider } from './context/ThemeContext'
import { ConnectionProvider } from './context/ConnectionContext'
import DashboardLayout from './layouts/DashboardLayout'
import Dashboard from './pages/Dashboard'
import HealthAnalytics from './pages/HealthAnalytics'
import EnvironmentalIntelligence from './pages/EnvironmentalIntelligence'
import AIRiskAnalysis from './pages/AIRiskAnalysis'
import ScenarioSimulator from './pages/ScenarioSimulator'
import Alerts from './pages/Alerts'

export default function App() {
  return (
    <ThemeProvider>
      <ConnectionProvider>
        <DashboardLayout>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/health" element={<HealthAnalytics />} />
            <Route path="/environment" element={<EnvironmentalIntelligence />} />
            <Route path="/risk" element={<AIRiskAnalysis />} />
            <Route path="/simulator" element={<ScenarioSimulator />} />
            <Route path="/alerts" element={<Alerts />} />
          </Routes>
        </DashboardLayout>
      </ConnectionProvider>
    </ThemeProvider>
  )
}
