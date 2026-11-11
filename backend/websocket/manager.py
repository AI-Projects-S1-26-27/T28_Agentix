from typing import Any
from fastapi import WebSocket
from fastapi.encoders import jsonable_encoder
from starlette.websockets import WebSocketDisconnect


class ConnectionManager:
    """Track dashboard clients and broadcast validated reading and risk payloads."""

    def __init__(self) -> None:
        self.active_connections: set[WebSocket] = set()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.add(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        self.active_connections.discard(websocket)

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
