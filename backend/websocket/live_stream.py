import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.websocket.manager import ConnectionManager

logger = logging.getLogger(__name__)

router = APIRouter(tags=["stream"])
manager = ConnectionManager()


@router.websocket("/ws/readings")
@router.websocket("/ws/events")
async def readings_stream(websocket: WebSocket) -> None:
    await manager.connect(websocket)
    try:
        while True:
            raw_text = await websocket.receive_text()
            # Handle client control messages
            try:
                msg = json.loads(raw_text)
                if isinstance(msg, dict):
                    msg_type = msg.get("type")
                    if msg_type == "ping":
                        await manager.send_personal(websocket, {"type": "pong"})
                    elif msg_type == "status":
                        await manager.send_personal(
                            websocket,
                            {
                                "type": "status_response",
                                "active_connections": manager.connection_count,
                            },
                        )
            except json.JSONDecodeError:
                if raw_text.strip().lower() == "ping":
                    await manager.send_personal(websocket, {"type": "pong"})
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except RuntimeError:
        manager.disconnect(websocket)
        raise
