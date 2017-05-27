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

def test_peak_next_state():
    pass