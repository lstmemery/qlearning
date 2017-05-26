from qlearning_numpy import *

def test_q_learning_size():
    assert state_grid.shape == (6, 9)

def test_make_transition_matrix():
    input_matrix = np.array([-1, 1],
                            [0, 0])
    output_matrix = np.array([[-1, -1, -1, -1],
                              [-1, -1, -1,  0],
                              [-1, -1, -1, -1],
                              [-1,  0,  1, -1]])
    assert make_transition_matrix(input_matrix) == output_matrix