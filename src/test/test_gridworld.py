import pytest

from src.gridworld import GridWorld, GridAgent


def test_grid_world():
    grid_world = GridWorld()
    assert grid_world.size == 46


def test_agent_in_start():
    gridworld = GridWorld()
    assert isinstance(gridworld.grid[5][3], GridAgent)


def test_agent_moves():
    gridworld = GridWorld()
    gridworld.agent.move("up")
    assert isinstance(gridworld.grid[4][3], GridAgent)
    gridworld.agent.move("down")
    assert isinstance(gridworld.grid[5][3], GridAgent)
    gridworld.agent.move("down")
    assert isinstance(gridworld.grid[5][3], GridAgent)


def test_agent_does_not_pass_impassible():
    gridworld = GridWorld()
    gridworld.agent.move("up")
    assert isinstance(gridworld.grid[4][3], GridAgent)
    gridworld.agent.move("up")
    assert isinstance(gridworld.grid[4][3], GridAgent)