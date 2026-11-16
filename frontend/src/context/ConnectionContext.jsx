import { createContext, useContext, useState } from 'react'

const ConnectionContext = createContext()
export const useConnection = () => useContext(ConnectionContext)

export function ConnectionProvider({ children }) {
  const [isConnected, setIsConnected] = useState(false)
  const [lastHeartbeat, setLastHeartbeat] = useState(null)
  const [connectionError, setConnectionError] = useState(null)
  // latestReading is set by App-level WebSocket handler so all pages get
  // real-time updates without each page owning its own WS connection
  const [latestReading, setLatestReading] = useState(null)

  return (
    <ConnectionContext.Provider value={{
      isConnected, setIsConnected,
      lastHeartbeat, setLastHeartbeat,
      connectionError, setConnectionError,
      latestReading, setLatestReading,
    }}>
      {children}
    </ConnectionContext.Provider>
  )
}
