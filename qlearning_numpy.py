import numpy as np

state_grid = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 1],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [-1, -1, -1, -1, -1, -1, -1, -1, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0]])

#TODO: Don;t forget about the transition at 1000 iterations







def make_transition_matrix(matrix):
    padded_matrix = np.pad(matrix, pad_width=1, mode="constant", constant_values=-1)

    up = np.roll(padded_matrix, shift=1, axis=0)[1:-1, 1:-1].flatten()
    right = np.roll(padded_matrix, shift=-1, axis=1)[1:-1, 1:-1].flatten()
    down = np.roll(padded_matrix, shift=-1, axis=0)[1:-1, 1:-1].flatten()
    left = np.roll(padded_matrix, shift=1, axis=1)[1:-1, 1:-1].flatten()

    transition_matrix = np.stack((up, right, down, left), axis=0).T

    return transition_matrix

def index_1d(row, col):
    pass

def qlearning(grid, episodes):
    r_matrix = make_transition_matrix(state_grid)
    q_matrix = np.zeros_like(r_matrix)
    for episode in episodes:
        pass



if __name__ == '__main__':
    print(make_transition_matrix(state_grid))
