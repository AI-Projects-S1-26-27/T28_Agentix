import { createContext, useContext, useState } from 'react'

const ConnectionContext = createContext()
export const useConnection = () => useContext(ConnectionContext)

export function ConnectionProvider({ children }) {
  const [isConnected, setIsConnected] = useState(false)
  const [lastHeartbeat, setLastHeartbeat] = useState(null)
  const [connectionError, setConnectionError] = useState(null)

  return (
    <ConnectionContext.Provider value={{ isConnected, setIsConnected, lastHeartbeat, setLastHeartbeat, connectionError, setConnectionError }}>
      {children}
    </ConnectionContext.Provider>
  )
}
