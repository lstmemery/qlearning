import numpy as np
from random import random, randint

state_grid = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 1],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [-1, -1, -1, -1, -1, -1, -1, -1, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0]])

#TODO: Don't forget about the transition at 1000 iterations


def make_transition_matrix(matrix):
    padded_matrix = np.pad(matrix, pad_width=1, mode="constant", constant_values=-1)

    up = np.roll(padded_matrix, shift=1, axis=0)[1:-1, 1:-1].flatten()
    right = np.roll(padded_matrix, shift=-1, axis=1)[1:-1, 1:-1].flatten()
    down = np.roll(padded_matrix, shift=-1, axis=0)[1:-1, 1:-1].flatten()
    left = np.roll(padded_matrix, shift=1, axis=1)[1:-1, 1:-1].flatten()

    transition_matrix = np.stack((up, right, down, left), axis=0).T

    return transition_matrix

def index_1d(row, col):
    max_cols = 9
    return row * max_cols + col

def reverse_index_1d(state):
    max_cols = 9
    row = state / max_cols
    col = state % max_cols
    return row, col

def peek_next_state(state, action):
    pass

def peek_reward(state, action):
    pass

def qlearning(grid, episodes, epsilon):
    r_matrix = make_transition_matrix(grid)
    # Initialize Q
    q_matrix = np.zeros_like(r_matrix)
    # For each episode
    for episode in episodes:
        #Initialize s
        state = index_1d(5, 3)
        # Repeat (for each step in an episode)
        while state != index_1d(0, 8):
            # With probability \epsilon choose random action
            if random() < epsilon:
                action = randint(0, 3)
            # With probability 1 - \epsilon choose argmax Q
            else:
                action = np.argmax(q_matrix[state])
        # Take action a, observe r (reward?), s'

        # Update Q
        # Update state
    # Until s is terminal



if __name__ == '__main__':
    print(make_transition_matrix(state_grid))
