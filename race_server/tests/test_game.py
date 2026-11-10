from app.game import CODE_ALPHABET, CODE_LENGTH, RoomManager
from tests.conftest import shortest_path


def make_room(interval=0.0):
    manager = RoomManager(rows=6, cols=6, min_move_interval=interval, seed=3)
    return manager, manager.create_room()


def test_race_starts_when_the_second_player_joins():
    _, room = make_room()
    p1 = room.add_player(now=0)
    assert p1.id == 1 and not room.started
    p2 = room.add_player(now=1)
    assert p2.id == 2 and room.started
    assert p1.position == p2.position == room.maze.start


def test_room_holds_at_most_two_players():
    _, room = make_room()
    room.add_player(0)
    room.add_player(0)
    assert room.add_player(0) is None


def test_moves_are_rejected_before_the_race_starts():
    _, room = make_room()
    room.add_player(0)
    assert (
        room.move(1, shortest_path(room.maze, room.maze.start, room.maze.cheese)[0], now=1)
        == "not_started"
    )


def test_moves_into_walls_are_rejected():
    _, room = make_room()
    room.add_player(0)
    room.add_player(0)
    row, col = room.maze.start
    wall_to_direction = {"top": "up", "bottom": "down", "left": "left", "right": "right"}
    blocked = next(
        wall_to_direction[w] for w, closed in room.maze.walls[row][col].items() if closed
    )
    assert room.move(1, blocked, now=1) == "wall"
    assert room.players[1].position == room.maze.start
    assert room.move(1, "sideways", now=1) == "invalid_direction"


def test_moves_faster_than_the_limit_are_rejected():
    _, room = make_room(interval=0.1)
    room.add_player(0)
    room.add_player(0)
    path = shortest_path(room.maze, room.maze.start, room.maze.cheese)
    assert room.move(1, path[0], now=1.0) == "ok"
    assert room.move(1, path[1], now=1.05) == "too_fast"
    assert room.move(1, path[1], now=1.2) == "ok"


def test_first_player_to_the_cheese_wins_and_the_race_ends():
    _, room = make_room()
    room.add_player(0)
    room.add_player(0)
    path = shortest_path(room.maze, room.maze.start, room.maze.cheese)
    for i, direction in enumerate(path):
        assert room.move(2, direction, now=i + 1) == "ok"
    assert room.winner == 2
    assert room.state()["winner"] == 2
    assert room.move(1, path[0], now=100) == "finished"


def test_empty_rooms_are_discarded_and_codes_are_readable():
    manager, room = make_room()
    assert len(room.code) == CODE_LENGTH and set(room.code) <= set(CODE_ALPHABET)
    assert manager.get(room.code.lower()) is room
    room.add_player(0)
    room.remove_player(1)
    manager.discard_if_empty(room.code)
    assert manager.get(room.code) is None
