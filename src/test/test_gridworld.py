import pytest

from src.gridworld import Gridworld


def test_grid_world():
    grid_world = Gridworld()
    assert grid_world.size == 46

def test_maze_agent_in_start():
    gridworld = Gridworld()
    assert isinstance(gridworld.grid[5][3], MazeAgent)