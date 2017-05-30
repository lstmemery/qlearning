import pytest

from src.qlearning import *


def test_make_transition_matrix():
    test_grid = np.array([[-1, 1],
                          [0, 0]])
    transition_matrix = make_transition_matrix(test_grid)
    assert transition_matrix.shape == (4, 4)
    assert transition_matrix.all() == np.array([[-1, 1, 0, -1],
                                                [-1, -1, 0, -1],
                                                [-1, 0, -1, -1],
                                                [1, -1, -1, 0]]).all()


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
    q = np.zeros_like(grid).astype(float)
    state = index_1d(1, 8)
    action = 0
    alpha = 0.1
    gamma = 0.95
    reward = peek_reward(grid, state, action)
    next_state = peek_next_state(grid, state, action)
    updated_q = update_q(q, state, action, alpha, gamma, reward, next_state)
    q[state, action] = updated_q
    assert q[state, action] == alpha


# noinspection PyTypeChecker
def test_q_learning():
    """This is an integration test. It determines whether there was a significant decrease in steps until
    completion.
    """
    _, step_list = qlearning(state_grid, 1000, epsilon=0.05, alpha=0.5, gamma=0.95, updated_grid=updated_grid)
    average_first_5 = sum(step_list[:5]) / len(step_list[:5])
    average_last_5 = sum(step_list[-5:]) / len(step_list[-5:])
    assert average_first_5 > 4 * average_last_5

@pytest.fixture()
def grid():
    return make_transition_matrix(state_grid)
