import { useEffect, useRef, useCallback } from 'react'
import { useConnection } from '../context/ConnectionContext'
import { HealthWebSocket } from '../services/websocket'

/**
 * Mounts a persistent WebSocket connection for the lifetime of the app.
 * Updates ConnectionContext on open/close, calls onReading for each data frame.
 *
 * Backend sends: { type: "reading" | "risk" | "alert", data: {...} }
 * We unwrap `data` and dispatch to the appropriate handler.
 */
export function useWebSocket(onReading, onRisk, onAlert) {
  const { setIsConnected, setLastHeartbeat, setConnectionError } = useConnection()
  const wsRef = useRef(null)
  const onReadingRef = useRef(onReading)
  const onRiskRef    = useRef(onRisk)
  const onAlertRef   = useRef(onAlert)

  useEffect(() => { onReadingRef.current = onReading }, [onReading])
  useEffect(() => { onRiskRef.current = onRisk },       [onRisk])
  useEffect(() => { onAlertRef.current = onAlert },     [onAlert])

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
      onMessage: (msg) => {
        setLastHeartbeat(new Date())
        // Backend wraps messages: { type, data }
        const type    = msg?.type
        const payload = msg?.data ?? msg   // fallback: treat flat object as reading

        if (type === 'reading' || (!type && payload?.heart_rate !== undefined)) {
          onReadingRef.current?.(payload)
        } else if (type === 'risk') {
          onRiskRef.current?.(payload)
        } else if (type === 'alert') {
          onAlertRef.current?.(payload)
        } else if (type && type !== 'pong') {
          // Unknown typed message — pass raw to reading handler as fallback
          onReadingRef.current?.(payload)
        }
      },
    })

    wsRef.current = ws
    ws.connect()

    return () => ws.disconnect()
  }, [setIsConnected, setLastHeartbeat, setConnectionError])

  return wsRef
}
