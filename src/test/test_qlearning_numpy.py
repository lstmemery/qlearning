from qlearning_numpy import *

def test_q_learning_size():
    assert make_transition_matrix(state_grid).shape == (46, 4)

def test_index_1d():
    assert index_1d(5, 3) == 48