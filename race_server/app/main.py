# fastapi app: endpoints to create and inspect rooms, plus a websocket per player
# gaze inference stays in each player's browser. the browser only sends moves
# (up/down/left/right) here. this server is the referee: it owns the shared
# maze, checks every move, and sends the updated positions to both players

import json
import logging
import sys
import time

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app.config import Settings
from app.game import RoomManager

logger = logging.getLogger("race")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def log_event(event: str, **fields: object) -> None:
    logger.info(json.dumps({"severity": "INFO", "event": event, **fields}))


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    app = FastAPI(title="Gaze Maze race backend", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.allowed_origins),
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    manager = RoomManager(
        rows=settings.maze_rows,
        cols=settings.maze_cols,
        min_move_interval=settings.min_move_interval_ms / 1000,
        seed=settings.seed,
    )
    # connections[room code][player id] = that player's WebSocket
    connections: dict[str, dict[int, WebSocket]] = {}
    app.state.manager = manager

    async def broadcast(code: str, message: dict) -> None:
        for socket in list(connections.get(code, {}).values()):
            try:
                await socket.send_json(message)
            except Exception:  # a socket that just closed; its handler cleans up
                pass

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "rooms": len(manager.rooms)}

    @app.post("/rooms", status_code=201)
    def create_room() -> dict:
        room = manager.create_room()
        log_event("room_created", room=room.code)
        return {"code": room.code}

    @app.get("/rooms/{code}")
    def get_room(code: str) -> dict:
        room = manager.get(code)
        if room is None:
            raise HTTPException(status_code=404, detail="room not found")
        return {
            "code": room.code,
            "players": len(room.players),
            "started": room.started,
            "winner": room.winner,
        }

    @app.websocket("/rooms/{code}/ws")
    async def play(websocket: WebSocket, code: str) -> None:
        await websocket.accept()
        room = manager.get(code)
        if room is None:
            await websocket.send_json({"type": "error", "reason": "room_not_found"})
            await websocket.close(code=4404)
            return
        player = room.add_player(time.monotonic())
        if player is None:
            await websocket.send_json({"type": "error", "reason": "room_full"})
            await websocket.close(code=4409)
            return

        connections.setdefault(room.code, {})[player.id] = websocket
        log_event("player_joined", room=room.code, player=player.id, players=len(room.players))
        await websocket.send_json(
            {"type": "welcome", "player": player.id, "maze": room.maze.to_json()}
        )
        await broadcast(room.code, room.state())
        if room.started and len(room.players) == 2:
            log_event("race_started", room=room.code)

        rejected = 0
        try:
            while True:
                try:
                    message = json.loads(await websocket.receive_text())
                except json.JSONDecodeError:
                    await websocket.send_json({"type": "rejected", "reason": "invalid_json"})
                    continue
                if not isinstance(message, dict) or message.get("type") != "move":
                    await websocket.send_json({"type": "rejected", "reason": "unknown_message"})
                    continue
                had_winner = room.winner is not None
                result = room.move(player.id, str(message.get("direction")), time.monotonic())
                if result != "ok":
                    rejected += 1
                    await websocket.send_json({"type": "rejected", "reason": result})
                    continue
                await broadcast(room.code, room.state())
                if not had_winner and room.winner is not None:
                    duration = round((room.finished_at or 0) - (room.started_at or 0), 2)
                    log_event("race_finished", room=room.code, winner=room.winner, seconds=duration)
        except WebSocketDisconnect:
            pass
        finally:
            connections.get(room.code, {}).pop(player.id, None)
            room.remove_player(player.id)
            log_event("player_left", room=room.code, player=player.id, rejected_moves=rejected)
            await broadcast(room.code, {"type": "opponent_left"})
            if not room.players:
                connections.pop(room.code, None)
                manager.discard_if_empty(room.code)

    return app


app = create_app()
