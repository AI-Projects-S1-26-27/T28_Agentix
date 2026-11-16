import { useEffect, useRef, useCallback } from 'react'
import { useConnection } from '../context/ConnectionContext'
import { HealthWebSocket } from '../services/websocket'

/**
 * Mounts a persistent WebSocket connection for the lifetime of the app.
 * Updates ConnectionContext on open/close, calls onReading for each data frame.
 */
export function useWebSocket(onReading) {
  const { setIsConnected, setLastHeartbeat, setConnectionError } = useConnection()
  const wsRef = useRef(null)
  const onReadingRef = useRef(onReading)

  // Keep callback ref fresh without re-mounting the socket
  useEffect(() => { onReadingRef.current = onReading }, [onReading])

  useEffect(() => {
    const ws = new HealthWebSocket({
      onOpen: () => {
        setIsConnected(true)
        setConnectionError(null)
        setLastHeartbeat(new Date())
      },
      onClose: () => {
        setIsConnected(false)
      },
      onError: (err) => {
        setConnectionError(err?.message || 'Connection error')
        setIsConnected(false)
      },
      onMessage: (data) => {
        setLastHeartbeat(new Date())
        onReadingRef.current?.(data)
      },
    })

    wsRef.current = ws
    ws.connect()

    return () => ws.disconnect()
  }, [setIsConnected, setLastHeartbeat, setConnectionError])

  return wsRef
}
