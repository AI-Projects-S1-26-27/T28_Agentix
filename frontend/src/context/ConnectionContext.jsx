import { createContext, useContext, useState, useEffect, useRef } from 'react'

const ConnectionContext = createContext()
export const useConnection = () => useContext(ConnectionContext)

// Grace period before showing "Disconnected" in the UI.
// Hides the brief flicker during the 3s auto-reconnect window.
const DISCONNECT_GRACE_MS = 4000

export function ConnectionProvider({ children }) {
  const [isConnected, setIsConnected] = useState(false)
  // displayConnected: what the UI actually shows — only flips false after grace period
  const [displayConnected, setDisplayConnected] = useState(false)
  const [lastHeartbeat, setLastHeartbeat] = useState(null)
  const [connectionError, setConnectionError] = useState(null)
  const [latestReading, setLatestReading] = useState(null)
  const disconnectTimer = useRef(null)

  useEffect(() => {
    if (isConnected) {
      // Connected immediately — cancel any pending "go offline" timer
      clearTimeout(disconnectTimer.current)
      setDisplayConnected(true)
    } else {
      // Disconnected — wait grace period before updating the UI
      disconnectTimer.current = setTimeout(() => {
        setDisplayConnected(false)
      }, DISCONNECT_GRACE_MS)
    }
    return () => clearTimeout(disconnectTimer.current)
  }, [isConnected])

  return (
    <ConnectionContext.Provider value={{
      isConnected: displayConnected, setIsConnected,
      lastHeartbeat, setLastHeartbeat,
      connectionError, setConnectionError,
      latestReading, setLatestReading,
    }}>
      {children}
    </ConnectionContext.Provider>
  )
}
