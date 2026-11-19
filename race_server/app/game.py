# this creates the rules and shared state for online races (rooms, players, moves, winner)
# separate from the solo maze in ui/, which runs fully in the browser so it works offline

import random
import secrets
import string
from dataclasses import dataclass, field

from app.maze import DIRECTIONS, Maze, Position, generate_maze

MAX_PLAYERS = 2

# randomly generating letters for unique code for people to join
CODE_ALPHABET = "".join(c for c in string.ascii_uppercase if c not in "ILO")
CODE_LENGTH = 4


@dataclass
class Player:
    id: int
    position: Position
    last_move_at: float = float("-inf")


@dataclass
class Room:
    code: str
    maze: Maze
    min_move_interval: float
    players: dict[int, Player] = field(default_factory=dict)
    started_at: float | None = None
    winner: int | None = None
    finished_at: float | None = None

    @property
    def started(self) -> bool:
        return self.started_at is not None

    @property
    def is_full(self) -> bool:
        return len(self.players) >= MAX_PLAYERS

    def add_player(self, now: float) -> Player | None:
        # room starts as soon as another player joins
        # we can work out these kinks and UI formalities in MS4
        if self.is_full:
            return None
        player_id = next(i for i in range(1, MAX_PLAYERS + 1) if i not in self.players)
        player = Player(id=player_id, position=self.maze.start)
        self.players[player_id] = player
        if self.is_full and not self.started:
            self.started_at = now
        return player

    def remove_player(self, player_id: int) -> None:
        self.players.pop(player_id, None)

    def move(self, player_id: int, direction: str, now: float) -> str:
        player = self.players.get(player_id)
        if player is None:
            return "unknown_player"
        if direction not in DIRECTIONS:
            return "invalid_direction"
        if not self.started:
            return "not_started"
        if self.winner is not None:
            return "finished"
        if now - player.last_move_at < self.min_move_interval:
            return "too_fast"
        if not self.maze.can_move(player.position, direction):
            return "wall"

        d_row, d_col, _, _ = DIRECTIONS[direction]
        player.position = (player.position[0] + d_row, player.position[1] + d_col)
        player.last_move_at = now
        if player.position == self.maze.cheese:
            self.winner = player_id
            self.finished_at = now
        return "ok"

    def state(self) -> dict:
        # game state that both players see
        return {
            "type": "state",
            "started": self.started,
            "players": {
                str(p.id): {"row": p.position[0], "col": p.position[1]}
                for p in self.players.values()
            },
            "winner": self.winner,
        }


class RoomManager:
    # keeps every active room in memory
    # rooms live in this process only so the service must run a singla instance

    def __init__(self, rows: int, cols: int, min_move_interval: float, seed: int | None = None):
        self.rows = rows
        self.cols = cols
        self.min_move_interval = min_move_interval
        self.rooms: dict[str, Room] = {}
        self._rng = random.Random(seed)

    def create_room(self) -> Room:
        code = self._new_code()
        room = Room(
            code=code,
            maze=generate_maze(self.rows, self.cols, self._rng),
            min_move_interval=self.min_move_interval,
        )
        self.rooms[code] = room
        return room

    def get(self, code: str) -> Room | None:
        return self.rooms.get(code.upper())

    def discard_if_empty(self, code: str) -> None:
        room = self.rooms.get(code)
        if room is not None and not room.players:
            del self.rooms[code]

    def _new_code(self) -> str:
        while True:
            code = "".join(secrets.choice(CODE_ALPHABET) for _ in range(CODE_LENGTH))
            if code not in self.rooms:
                return code
