from collections import deque

from app.maze import DIRECTIONS, Maze, Position


def shortest_path(maze: Maze, start: Position, goal: Position) -> list[str]:
    """Directions along the (unique) path between two cells, found by BFS."""
    came_from: dict[Position, tuple[Position, str] | None] = {start: None}
    queue = deque([start])
    while queue:
        cell = queue.popleft()
        if cell == goal:
            break
        for name, (d_row, d_col, _, _) in DIRECTIONS.items():
            if maze.can_move(cell, name):
                nxt = (cell[0] + d_row, cell[1] + d_col)
                if nxt not in came_from:
                    came_from[nxt] = (cell, name)
                    queue.append(nxt)
    path: list[str] = []
    cell = goal
    while came_from[cell] is not None:
        prev, name = came_from[cell]
        path.append(name)
        cell = prev
    return path[::-1]
