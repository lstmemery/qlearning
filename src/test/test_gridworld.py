import pytest

from src.gridworld import GridWorld, GridAgent


def test_grid_world():
    grid_world = GridWorld()
    assert grid_world.size == 46


def test_agent_in_start():
    gridworld = GridWorld()
    assert isinstance(gridworld.grid[5][3], GridAgent)