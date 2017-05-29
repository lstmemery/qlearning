# coding=utf-8
import numpy as np

from random import random, randint

state_grid = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 1],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [-1, -1, -1, -1, -1, -1, -1, -1, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0]])

updated_grid = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 1],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, -1, -1, -1, -1, -1, -1, -1, -1],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0]])


def make_transition_matrix(matrix):
    """Derive a transition matrix from a 2D array representing a grid world.

     By convention, the four transitions (up, right, down, and left) are represented by columns 0 to 3, respectively.
     The value -1 represents an impassible value. The value 1 represents the reward value.

    Parameters
    ----------
    matrix : ndarray of ints
        A 2D array representing all possible transitions.

    Returns
    -------
    transition_matrix : ndarray of ints
        An N X 4 2D array where N is the total size of matrix.

    Notes
    -----
    It would be possible to remove the impassible values from the transition matrix but I chose not to for ease of
    indexing.
    """
    padded_matrix = np.pad(matrix, pad_width=1, mode="constant", constant_values=-1)

    up = np.roll(padded_matrix, shift=1, axis=0)[1:-1, 1:-1].flatten()
    right = np.roll(padded_matrix, shift=-1, axis=1)[1:-1, 1:-1].flatten()
    down = np.roll(padded_matrix, shift=-1, axis=0)[1:-1, 1:-1].flatten()
    left = np.roll(padded_matrix, shift=1, axis=1)[1:-1, 1:-1].flatten()

    transition_matrix = np.stack((up, right, down, left), axis=0).T

    return transition_matrix


def index_1d(row, col):
    """For a given row and column in a grid world, return the row that cell would appear on in the transition matrix.

    Parameters
    ----------
    row : int
        The row number. Indexing starts at 0.
    col : int
        The column number. Indexing starts at 0.

    Returns
    -------
    transition_row : int
        The row on the transition matrix the row and column would appear on.
    """
    max_cols = 9
    return row * max_cols + col


def reverse_index_1d(state):
    """The inverse of index_1d. From an row number in the transition matrix return the row and column in the grid world.

    Parameters
    ----------
    state : int
        An integer representing the row in the transition matrix.

    Returns
    -------
    row : int
        The row in the grid world.
    column : int
        The column in the grid world.\
    """
    max_cols = 9
    row = state / max_cols
    col = state % max_cols
    return row, col


def peek_next_state(grid, state, action):
    """For a given grid, state, and action return the next state in the transition matrix.

    If the next state is less than 0, stay in the same state.

    Parameters
    ----------
    grid : ndarray of ints
        The transition matrix.
    state : int
        The current state of the transition matrix (represented as a row number)
    action : int
        The action being observed. 0 = up. 1 = right. 2 = down. 3 = right.

    Returns
    -------
    next_state : int
        The next state in the transition matrix.
    """
    assert state <= grid.shape[0], "State larger than grid size"
    assert action <= grid.shape[1], "Action larger than grid size"

    if grid[state, action] < 0:
        return state
    else:
        row, col = reverse_index_1d(state)
        if action == 0:
            return index_1d(row - 1, col)
        elif action == 1:
            return index_1d(row, col + 1)
        elif action == 2:
            return index_1d(row + 1, col)
        elif action == 3:
            return index_1d(row, col - 1)
        else:
            raise ValueError("Action cannot be greater than 3")


def peek_reward(grid, state, action):
    """For a given grid, state, and action return the reward in the transition matrix.

    Parameters
    ----------
    grid : ndarray of ints
        The transition matrix.
    state : int
        The current state of the transition matrix (represented as a row number)
    action : int
        The action being observed. 0 = up. 1 = right. 2 = down. 3 = right.

    Returns
    -------
    reward : int
        The reward in the transition matrix
    """
    return grid[state, action]


def update_q(q, state, action, alpha, gamma, reward, next_state):
    """Update the Q matrix according to formula:

    .. math::Q(s, a) ← Q(s, a) + α r + γ max a Q(s 0 , a) − Q(s, a)


    Parameters
    ----------
    q : ndarray of floats
        The 2D matrix the represents the Q function.
    state : int
        The current state of the transition matrix.
    action : int
        The action taken.
    alpha : float
        The learning rate. This determines how much each step is weighted.
    gamma : float
        The discount factor. The weight that the algorithm will consider future rewards. gamma = 0 is entirely greedy.
        Values higher than 1 can cause the algorithm to diverge.[1]
    reward : int
        The reward in the transition matrix
    next_state : int
        The next state of the transition matrix.

    Returns
    -------
    q_value : The value the Q matrix will be updated with

    References
    ----------
    1. Q-learning. In: Wikipedia [Internet]. 2017 [cited 2017 May 29]. Available from: https://en.wikipedia.org/w/index.php?title=Q-learning&oldid=762556833

    """
    max_next_step = max(q[next_state, :])
    next_q = q[state, action] + alpha * (reward + gamma * max_next_step - q[state, action])
    return next_q


def qlearning(grid, episodes, epsilon, alpha, gamma, updated_grid):
    r_matrix = make_transition_matrix(grid)
    # Initialize Q
    q_matrix = np.zeros_like(r_matrix).astype(float)
    # For each episode
    steps = 0
    steps_to_goal_list = []
    for _ in range(episodes):
        # Initialize s
        state = index_1d(5, 3)

        steps_to_goal = 0

        # Repeat (for each step in an episode)
        while state != index_1d(0, 8):
            if steps == 1000:
                r_matrix = make_transition_matrix(updated_grid)
            # With probability \epsilon choose random action
            action = get_epsilon_greedy_action(epsilon, q_matrix, state)
            # Take action a, observe r, s'
            reward = peek_reward(r_matrix, state, action)
            next_state = peek_next_state(r_matrix, state, action)
            # Update Q
            q_matrix[state, action] = update_q(q_matrix, state, action, alpha, gamma, reward, next_state)
            # Update state
            state = next_state
            steps += 1
            steps_to_goal += 1

        steps_to_goal_list.append(steps_to_goal)
    # Until s is terminal

    return q_matrix, steps_to_goal_list


def get_epsilon_greedy_action(epsilon, q_matrix, state):
    if random() < epsilon:
        action = randint(0, 3)
    # With probability 1 - \epsilon choose argmax Q
    else:
        action = np.random.choice(np.where(q_matrix[state] == q_matrix[state].max())[0])
    return action


if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    q, iterations = qlearning(state_grid, 1000, epsilon=0.05, alpha=0.5, gamma=0.95, updated_grid=updated_grid)
    print(q)
    print(iterations)
    print(sum(iterations) / len(iterations))
