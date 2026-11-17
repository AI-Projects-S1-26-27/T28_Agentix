import logging
from typing import Any, Dict, Optional, Set

from fastapi import WebSocket
from fastapi.encoders import jsonable_encoder
from starlette.websockets import WebSocketDisconnect

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Track dashboard clients and broadcast validated reading, risk, and alert payloads."""

    def __init__(self) -> None:
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.add(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        self.active_connections.discard(websocket)

    @property
    def connection_count(self) -> int:
        return len(self.active_connections)

    async def send_personal(self, websocket: WebSocket, payload: Any) -> None:
        safe_payload = jsonable_encoder(payload)
        try:
            await websocket.send_json(safe_payload)
        except (WebSocketDisconnect, RuntimeError):
            self.disconnect(websocket)

    async def broadcast(self, payload: Any) -> None:
        safe_payload = jsonable_encoder(payload)
        disconnected: list[WebSocket] = []
        for websocket in tuple(self.active_connections):
            try:
                await websocket.send_json(safe_payload)
            except (WebSocketDisconnect, RuntimeError):
                disconnected.append(websocket)
        for websocket in disconnected:
            self.disconnect(websocket)

    async def broadcast_event(self, event_type: str, data: Any) -> None:
        payload = {
            "type": event_type,
            "data": data,
        }
        await self.broadcast(payload)
