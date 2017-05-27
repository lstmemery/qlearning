import pytest
from qlearning_numpy import *

def test_q_learning_size():
    assert make_transition_matrix(state_grid).shape == (54, 4)

def test_index_1d():
    assert index_1d(5, 3) == 48
    transition_matrix = make_transition_matrix(state_grid)
    assert transition_matrix[index_1d(0, 8), :].all() == np.array([-1, -1, 0, 0]).all()


def test_reverse_index_1d():
    indices = [(0, 5), (3, 6), (5, 8)]
    for index in indices:
        assert reverse_index_1d(index_1d(*index)) == index


def test_peek_next_state(grid):
    # up
    assert peek_next_state(grid, index_1d(5, 3), 0) == index_1d(4, 3)
    # right
    assert peek_next_state(grid, index_1d(5, 3), 1) == index_1d(5, 4)
    # down
    assert peek_next_state(grid, index_1d(5, 3), 2) == index_1d(5, 3)
    # left
    assert peek_next_state(grid, index_1d(5, 3), 3) == index_1d(5, 2)


def test_peek_reward(grid):
    # up
    assert peek_reward(grid, index_1d(5, 3), 0) == 0
    # right
    assert peek_reward(grid, index_1d(5, 3), 1) == 0
    # down
    assert peek_reward(grid, index_1d(5, 3), 2) == -1
    # left
    assert peek_reward(grid, index_1d(5, 3), 3) == 0


def test_update_q(grid):
    q = np.zeros_like(grid)
    state = index_1d(1, 8)
    action = 0
    alpha = 0.1
    gamma = 0.95
    updated_q = update_q(q, state, action, alpha, gamma)
    reward = peek_reward(grid, state, action)
    next_state = peek_next_state(grid, state, action)
    assert updated_q[state, action] == alpha * (reward + gamma * max(q[next_state, :]) - q[state, action])


@pytest.fixture()
def grid():
    return make_transition_matrix(state_grid)