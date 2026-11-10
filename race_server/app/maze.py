#randomly generates the race maze (same dfs algorithm as the solo game) and checks if a move hits a wall
#the server makes the maze so both players get the exact same one and nobody can send a fake easy maze

import random
from dataclasses import dataclass, field

Position = tuple[int, int]  # (row, col)

DIRECTIONS: dict[str, tuple[int, int, str, str]] = {
    # direction: (row step, col step, wall on this cell, wall on the next cell)
    "up": (-1, 0, "top", "bottom"),
    "down": (1, 0, "bottom", "top"),
    "left": (0, -1, "left", "right"),
    "right": (0, 1, "right", "left"),
}


@dataclass
class Maze:
    rows: int
    cols: int
    # walls[row][col] = {"top": bool, "right": bool, "bottom": bool, "left": bool}
    walls: list[list[dict[str, bool]]] = field(default_factory=list)
    start: Position = (0, 0)
    cheese: Position = (0, 0)

    def can_move(self, position: Position, direction: str) -> bool:
        """True if moving one cell in `direction` stays inside the maze and crosses no wall."""
        if direction not in DIRECTIONS:
            return False
        row, col = position
        d_row, d_col, wall, _ = DIRECTIONS[direction]
        if self.walls[row][col][wall]:
            return False
        next_row, next_col = row + d_row, col + d_col
        return 0 <= next_row < self.rows and 0 <= next_col < self.cols

    def to_json(self) -> dict:
        return {
            "rows": self.rows,
            "cols": self.cols,
            "walls": self.walls,
            "start": {"row": self.start[0], "col": self.start[1]},
            "cheese": {"row": self.cheese[0], "col": self.cheese[1]},
        }


def generate_maze(rows: int, cols: int, rng: random.Random | None = None) -> Maze:
    #each player gets the exact same maze
    if rows < 2 or cols < 2:
        raise ValueError(f"maze must be at least 2x2, got {rows}x{cols}")
    rng = rng or random.Random()
    walls = [
        [{"top": True, "right": True, "bottom": True, "left": True} for _ in range(cols)]
        for _ in range(rows)
    ]

    # Randomized depth-first search with backtracking, from the top-left cell.
    visited = [[False] * cols for _ in range(rows)]
    stack: list[Position] = [(0, 0)]
    visited[0][0] = True
    while stack:
        row, col = stack[-1]
        neighbors = []
        for name, (d_row, d_col, _, _) in DIRECTIONS.items():
            n_row, n_col = row + d_row, col + d_col
            if 0 <= n_row < rows and 0 <= n_col < cols and not visited[n_row][n_col]:
                neighbors.append((name, n_row, n_col))
        if not neighbors:
            stack.pop()
            continue
        name, n_row, n_col = rng.choice(neighbors)
        _, _, wall, opposite = DIRECTIONS[name]
        walls[row][col][wall] = False
        walls[n_row][n_col][opposite] = False
        visited[n_row][n_col] = True
        stack.append((n_row, n_col))

    # Start on one side, cheese on the opposite side (top or bottom independently).
    start_left = rng.random() < 0.5
    start = (0 if rng.random() < 0.5 else rows - 1, 0 if start_left else cols - 1)
    cheese = (0 if rng.random() < 0.5 else rows - 1, cols - 1 if start_left else 0)

    # Open an entrance and an exit on the outer edge, as in the UI.
    walls[start[0]][start[1]]["left" if start[1] == 0 else "right"] = False
    walls[cheese[0]][cheese[1]]["left" if cheese[1] == 0 else "right"] = False

    return Maze(rows=rows, cols=cols, walls=walls, start=start, cheese=cheese)
