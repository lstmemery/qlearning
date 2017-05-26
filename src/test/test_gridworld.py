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

def test_agent_rewarded_on_goal():
    gridworld = GridWorld()
    gridworld.agent.move("right")
    gridworld.agent.move("right")
    gridworld.agent.move("right")
    gridworld.agent.move("right")
    gridworld.agent.move("right")
    gridworld.agent.move("up")
    gridworld.agent.move("up")
    gridworld.agent.move("up")
    gridworld.agent.move("up")
    gridworld.agent.move("up")
    assert gridworld.agent.reward == 1
    assert isinstance(gridworld.grid[5][3], GridAgent)

def test_neighbours():
    gridworld = GridWorld()
    assert [location[0] for location in gridworld.agent.get_neighbours()] == ["left", "right", "up"]