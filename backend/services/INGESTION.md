# Ingestion Pipeline Architecture
- Ring buffer window: 60-tick sliding cache
- Lock-free memory buffer for WebSocket broadcast
- Asynchronous SQLite batch commit worker
