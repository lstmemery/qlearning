import pytest

from src.gridworld import Gridworld


def test_grid_world():
    grid_world = Gridworld()
    print(grid_world.size)
    assert grid_world.size == 46
