from qlearning_numpy import *

def test_q_learning_size():
    assert make_transition_matrix(state_grid).shape == (54, 4)

def test_index_1d():
    assert index_1d(5, 3) == 48
    transition_matrix = make_transition_matrix(state_grid)
    assert transition_matrix[index_1d(0, 8), :].all() == np.array([-1, -1, 0 , 0]).all()