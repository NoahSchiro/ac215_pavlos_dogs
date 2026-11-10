import random

import pytest

from app.maze import generate_maze
from tests.conftest import shortest_path


def test_every_cell_is_reachable_by_exactly_one_path():
    maze = generate_maze(15, 15, random.Random(0))
    # A perfect maze on R x C cells has exactly R*C - 1 open passages.
    passages = sum(
        (not maze.walls[r][c]["right"] and c + 1 < maze.cols)
        + (not maze.walls[r][c]["bottom"] and r + 1 < maze.rows)
        for r in range(maze.rows)
        for c in range(maze.cols)
    )
    assert passages == maze.rows * maze.cols - 1
    for r in range(maze.rows):
        for c in range(maze.cols):
            assert shortest_path(maze, maze.start, (r, c)) is not None


def test_start_and_cheese_are_corners_on_opposite_sides():
    for seed in range(20):
        maze = generate_maze(10, 10, random.Random(seed))
        assert maze.start[0] in (0, 9) and maze.cheese[0] in (0, 9)
        assert {maze.start[1], maze.cheese[1]} == {0, 9}


def test_walls_and_edges_block_moves():
    maze = generate_maze(5, 5, random.Random(1))
    row, col = maze.start
    # The entrance wall is open, but the edge of the maze still blocks the move.
    outward = "left" if col == 0 else "right"
    assert not maze.walls[row][col][outward]
    assert not maze.can_move(maze.start, outward)
    for direction, blocked in maze.walls[row][col].items():
        if blocked:
            assert not maze.can_move(maze.start, direction)
    assert not maze.can_move(maze.start, "diagonal")


def test_same_seed_gives_same_maze():
    a = generate_maze(8, 8, random.Random(42))
    b = generate_maze(8, 8, random.Random(42))
    assert a.to_json() == b.to_json()


def test_rejects_tiny_mazes():
    with pytest.raises(ValueError):
        generate_maze(1, 5)
