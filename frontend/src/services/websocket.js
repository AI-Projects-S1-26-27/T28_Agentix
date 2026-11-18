// Connect to Vite proxy (/ws) in dev, or directly to Render in production
const isLocal =
  typeof window !== 'undefined' &&
  (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1')

const WS_URL = isLocal
  ? `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws/readings`
  : 'wss://healthsensing.onrender.com/ws/readings'

const RECONNECT_DELAY_MS = 3000
const HEARTBEAT_INTERVAL_MS = 20000

export class HealthWebSocket {
  constructor({ onMessage, onOpen, onClose, onError } = {}) {
    this._onMessage = onMessage || (() => {})
    this._onOpen    = onOpen    || (() => {})
    this._onClose   = onClose   || (() => {})
    this._onError   = onError   || (() => {})
    this._ws        = null
    this._reconnectTimer = null
    this._heartbeatTimer = null
    this._stopped   = false
  }

  connect() {
    if (this._stopped) return
    try {
      this._ws = new WebSocket(WS_URL)

      this._ws.onopen = () => {
        this._onOpen()
        this._startHeartbeat()
      }

      this._ws.onmessage = (evt) => {
        try {
          const data = JSON.parse(evt.data)
          if (data?.type !== 'pong') this._onMessage(data)
        } catch {
          // ignore malformed frames
        }
      }

      this._ws.onclose = () => {
        this._stopHeartbeat()
        this._onClose()
        if (!this._stopped) {
          this._reconnectTimer = setTimeout(() => this.connect(), RECONNECT_DELAY_MS)
        }
      }

      this._ws.onerror = (err) => {
        this._onError(err)
        this._ws?.close()
      }
    } catch (err) {
      this._onError(err)
      if (!this._stopped) {
        this._reconnectTimer = setTimeout(() => this.connect(), RECONNECT_DELAY_MS)
      }
    }
  }

  disconnect() {
    this._stopped = true
    clearTimeout(this._reconnectTimer)
    this._stopHeartbeat()
    this._ws?.close()
  }

  _startHeartbeat() {
    this._heartbeatTimer = setInterval(() => {
      if (this._ws?.readyState === WebSocket.OPEN) {
        this._ws.send(JSON.stringify({ type: 'ping' }))
      }
    }, HEARTBEAT_INTERVAL_MS)
  }

  _stopHeartbeat() {
    clearInterval(this._heartbeatTimer)
  }
}
