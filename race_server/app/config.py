# race server settings config here

import os
from dataclasses import dataclass

# race maze size, in cells. server sends the maze to both players, so this
# decides what they see (the UI's MAZE_ROWS/MAZE_COLS only affect solo mode)
MAZE_ROWS = 15
MAZE_COLS = 15

# fastest a player may move, in milliseconds (anti-cheat). The UI sends a move
# every 160 ms (KEYBOARD_MOVE_INTERVAL_MS / GAZE_MOVE_INTERVAL_MS) so normal
# play never hits this limit.
MIN_MOVE_INTERVAL_MS = 100

# Websites allowed to call this server (CORS), comma-separated. "*" allows any;
# in production set it to the deployed UI's URL.
ALLOWED_ORIGINS = "*"


@dataclass(frozen=True)
class Settings:
    maze_rows: int = MAZE_ROWS
    maze_cols: int = MAZE_COLS
    min_move_interval_ms: int = MIN_MOVE_INTERVAL_MS
    allowed_origins: tuple[str, ...] = (ALLOWED_ORIGINS,)
    # fixed random seed for reproducible mazes
    seed: int | None = None

    @classmethod
    def from_env(cls) -> "Settings":
        # Defaults above, replaced by any matching environment variables
        origins = os.environ.get("ALLOWED_ORIGINS", ALLOWED_ORIGINS)
        return cls(
            maze_rows=int(os.environ.get("MAZE_ROWS", MAZE_ROWS)),
            maze_cols=int(os.environ.get("MAZE_COLS", MAZE_COLS)),
            min_move_interval_ms=int(os.environ.get("MIN_MOVE_INTERVAL_MS", MIN_MOVE_INTERVAL_MS)),
            allowed_origins=tuple(o.strip() for o in origins.split(",") if o.strip()),
        )
