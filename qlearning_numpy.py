import numpy as np


state_grid = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [-1, -1, -1, -1, -1, -1, -1, -1, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0]])



action_grid = np.zeros((6, 9))

def make_transition_matrix(matrix):
    transition_matrix = np.zeros((matrix.size, matrix.size)) - 1

    rows, columns = matrix.shape

    for row in range(0, rows):
        for column in range(0, columns):
            if row > 0 and matrix[row - 1, column] >= 0:
                transition_matrix[column * rows + row, column * rows + row - 1] = matrix[row - 1, column]

            if row < rows and matrix[row + 1, column] >= 0:
                transition_matrix[column * rows + row, column * rows + row + 1] = matrix[row + 1, column]

            if column > 0 and matrix[row, column - 1] >= 0:
                transition_matrix[column * rows + row, (column - 1) * rows + row] = matrix[row, column - 1]

            if column > 0 and matrix[row, column - 1] >= 0:
                transition_matrix[column * rows + row, (column - 1) * rows + row] = matrix[row, column - 1]

    print(transition_matrix)
    return transition_matrix

if __name__ == '__main__':
    print(action_grid)