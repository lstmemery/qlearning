import pytest
from src.qlearning import *

def test_get_best_neighbour(gridworld):
    action_grid = [[0 for _ in range(gridworld.x)] for _ in range(gridworld.y)]
    action_grid[5][2] = 100
    assert get_best_neighbour(gridworld, action_grid) == "left"


@pytest.fixture()
def gridworld():
    return GridWorld()