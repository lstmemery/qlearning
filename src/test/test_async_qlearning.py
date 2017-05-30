from random import seed

from src.async_qlearning import *


def test_async_q_learning():
    """This is an integration test. It determines whether there was a significant decrease in steps until
    completion.
    """
    _, step_list = async_qlearning(processes=2,
                                   epsilon=0.55,
                                   alpha=0.45,
                                   gamma=0.95,
                                   async_update=5,
                                   Tmax=10000)
    average_first_5 = sum(step_list[:5]) / len(step_list[:5])
    average_last_5 = sum(step_list[-5:]) / len(step_list[-5:])
    assert average_first_5 > 4 * average_last_5


def test_update_delta_q():
    local_q_matrix = np.array([[1, 0],
                               [0, 0]])
    state = 1
    action = 0
    gamma = 0.95
    reward = 1
    next_state = 0
    global_q_matrix = np.array([[0, 1],
                                [0, 0]])

    assert update_delta_q(local_q_matrix, state, action, gamma, reward, next_state, global_q_matrix) == 1.95


def test_get_async_epsilon_greedy_action():
    q_matrix = np.array([[0.1, 0.2],
                         [0.3, 0.4]])
    seed(0)
    state = 0
    epsilon = 0.01
    assert get_async_epsilon_greedy_action(epsilon, q_matrix, state) == 1


def test_qlearning_worker_step_queue():
    r_matrix = ql.make_transition_matrix(ql.state_grid)
    epsilon = 0.30
    gamma = 0.95
    async_update = 5
    T = Value('i', 0)
    Tmax = 1000
    alpha = 0.30

    manager = Manager()

    lock = Lock()
    step_queue = manager.Queue()

    raw_array = Array(ctypes.c_double, r_matrix.shape[0] * r_matrix.shape[1], lock=lock)
    shared_array = np.ctypeslib.as_array(raw_array.get_obj())
    global_q_matrix = shared_array.reshape(r_matrix.shape)

    qlearning_worker(r_matrix, epsilon, gamma, async_update, T, Tmax, alpha, global_q_matrix, step_queue,
                     raw_array)

    step_list = []

    while not step_queue.empty():
        step_list.append(step_queue.get())

    assert len(step_list) >= 1
