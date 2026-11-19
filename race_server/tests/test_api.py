import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.config import Settings
from app.main import create_app
from tests.conftest import shortest_path


@pytest.fixture
def client():
    app = create_app(Settings(maze_rows=4, maze_cols=4, min_move_interval_ms=0, seed=7))
    with TestClient(app) as c:
        yield c


def create_room(client) -> str:
    response = client.post("/rooms")
    assert response.status_code == 201
    return response.json()["code"]


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_create_and_inspect_a_room(client):
    code = create_room(client)
    body = client.get(f"/rooms/{code}").json()
    assert body == {"code": code, "players": 0, "started": False, "winner": None}
    assert client.get("/rooms/ZZZZ").status_code == 404


def test_two_players_race_and_both_see_every_move(client):
    code = create_room(client)
    with client.websocket_connect(f"/rooms/{code}/ws") as p1:
        welcome1 = p1.receive_json()
        assert welcome1["type"] == "welcome" and welcome1["player"] == 1
        assert p1.receive_json()["started"] is False

        with client.websocket_connect(f"/rooms/{code}/ws") as p2:
            welcome2 = p2.receive_json()
            assert welcome2["player"] == 2
            assert welcome2["maze"] == welcome1["maze"]  # same layout for both players
            assert p1.receive_json()["started"] is True
            assert p2.receive_json()["started"] is True

            # Rebuild the maze locally to find the path to the cheese.
            maze_json = welcome1["maze"]
            from app.maze import Maze

            maze = Maze(
                rows=maze_json["rows"],
                cols=maze_json["cols"],
                walls=maze_json["walls"],
                start=(maze_json["start"]["row"], maze_json["start"]["col"]),
                cheese=(maze_json["cheese"]["row"], maze_json["cheese"]["col"]),
            )
            path = shortest_path(maze, maze.start, maze.cheese)

            for direction in path:
                p1.send_json({"type": "move", "direction": direction})
                state1, state2 = p1.receive_json(), p2.receive_json()
                assert state1 == state2  # both players see the same state
            assert state1["winner"] == 1
            assert state1["players"]["1"] == maze_json["cheese"]

            p2.send_json({"type": "move", "direction": path[0]})
            assert p2.receive_json() == {"type": "rejected", "reason": "finished"}


def test_invalid_moves_are_rejected_without_disconnecting(client):
    code = create_room(client)
    with client.websocket_connect(f"/rooms/{code}/ws") as p1:
        p1.receive_json(), p1.receive_json()
        p1.send_json({"type": "move", "direction": "up"})
        assert p1.receive_json() == {"type": "rejected", "reason": "not_started"}
        p1.send_text("not json")
        assert p1.receive_json() == {"type": "rejected", "reason": "invalid_json"}
        p1.send_json({"type": "dance"})
        assert p1.receive_json() == {"type": "rejected", "reason": "unknown_message"}


def test_unknown_and_full_rooms_are_refused(client):
    with client.websocket_connect("/rooms/ZZZZ/ws") as ws:
        assert ws.receive_json() == {"type": "error", "reason": "room_not_found"}
        with pytest.raises(WebSocketDisconnect):
            ws.receive_json()

    code = create_room(client)
    with (
        client.websocket_connect(f"/rooms/{code}/ws"),
        client.websocket_connect(f"/rooms/{code}/ws"),
    ):
        with client.websocket_connect(f"/rooms/{code}/ws") as third:
            assert third.receive_json() == {"type": "error", "reason": "room_full"}


def test_opponent_is_told_when_the_other_player_leaves(client):
    code = create_room(client)
    with client.websocket_connect(f"/rooms/{code}/ws") as p1:
        p1.receive_json(), p1.receive_json()
        with client.websocket_connect(f"/rooms/{code}/ws") as p2:
            p2.receive_json(), p2.receive_json()
            p1.receive_json()  # state: race started
        assert p1.receive_json() == {"type": "opponent_left"}
    assert client.get(f"/rooms/{code}").status_code == 404  # room cleaned up
