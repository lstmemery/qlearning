# coding=utf-8
from multiprocessing import Process, Value, Array, Manager, Lock
import numpy as np
import src.qlearning as ql
import ctypes
from random import random, randint

updated_matrix = ql.make_transition_matrix(ql.updated_grid)


def update_delta_q(q, state, action, gamma, reward, next_state, global_q_state):
    """

    ∆Q(s, a) ← ∆Q(s, a) + r + γ max a Q(s 0 , a) − Q(s, a)

    Parameters
    ----------
    q
    state
    action
    gamma
    reward
    next_state
    global_q_state

    Returns
    -------

    """
    max_next_step = max(global_q_state[next_state, :])
    next_q = q[state, action] + (reward + (gamma * max_next_step) - global_q_state[state, action])
    return next_q


def qlearning_worker(r_matrix, epsilon, gamma, async_update, T, Tmax, alpha, global_q_matrix, step_queue, raw_array):
    # Initialize thread step count t <- 1 (Not sure why this starts at 1)
    t = 1
    last_reward = 1
    total_reward = 0
    delta_q_matrix = np.zeros_like(r_matrix).astype(float)

    start_state = ql.index_1d(5, 3)
    # Get initial state s
    state = start_state

    while T.value <= Tmax:
        if T.value > 1000:
            r_matrix = updated_matrix
        # Choose action from state s using \epsilon-greedy policy

        action = get_async_epsilon_greedy_action(epsilon, global_q_matrix, state, raw_array)
        # Take action a, observe r, s'
        reward = ql.peek_reward(r_matrix, state, action)
        next_state = ql.peek_next_state(r_matrix, state, action)
        updated_q = update_delta_q(delta_q_matrix, state, action, gamma, reward, next_state, global_q_matrix)
        # Accumulate updates:
        delta_q_matrix[state, action] += updated_q
        # s <- s'
        state = next_state
        # t <- t + 1 and T <- T + 1
        t += 1

        T.value += 1

        # if t % I_AsyncUpdate == 0 or s is terminal then
        if t % async_update == 0 or state == ql.index_1d(0, 8):
            # Perform async update
            with raw_array.get_lock():
                global_q_matrix[delta_q_matrix.nonzero()] += alpha * delta_q_matrix[delta_q_matrix.nonzero()]

            # clear updates \DeltaQ(s', a')
            delta_q_matrix = np.zeros_like(r_matrix).astype(float)
            if state == ql.index_1d(0, 8):
                step_queue.put(t - last_reward)
                last_reward = t
                state = start_state
                total_reward += 1


def get_async_epsilon_greedy_action(epsilon, q_matrix, state, raw_matrix):
    """Choose the next action in the q-learning algorithm.

    Parameters
    ----------
    epsilon : float
        The probability of choosing the next action randomly. Otherwise, choose the Q-optimal next step.
    q_matrix : ndarray of floats
        The 2D matrix representing the Q-function.
    state : int
        The current state, represented as a row number in the transition matrix.

    Returns
    -------
    action : int
        The next action to be taken, represented as a column number in the transition matrix.


    Notes
    -----
    If there are multiple q-optimal actions (i.e. a tie), this function selects randomly between the optimal values.

    """
    if random() < epsilon:
        action = randint(0, 3)
    # With probability 1 - \epsilon choose argmax Q
    else:
        with raw_matrix.get_lock():
            action = np.random.choice(np.where(q_matrix[state] == q_matrix[state].max())[0])
    return action


def async_manager(processes, epsilon, alpha, gamma, async_update, Tmax):
    # Assume global shared Q(s, a) function values, and counter T = 0
    T = Value('i', 0, lock=False)
    r_matrix = ql.make_transition_matrix(ql.state_grid)
    # Initialize global Q(s, a)

    manager = Manager()

    q = manager.Queue()
    lock = Lock()

    shared_array_base = Array(ctypes.c_double, r_matrix.shape[0] * r_matrix.shape[1], lock=lock)
    shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
    shared_array = shared_array.reshape(r_matrix.shape)

    step_queue = manager.Queue(2048)

    producer1 = Process(target=qlearning_worker, args=(r_matrix, epsilon, gamma,
                                                      async_update, T,
                                                      Tmax, alpha, shared_array, step_queue, shared_array_base))

    producer1.start()

    producer2 = Process(target=qlearning_worker, args=(r_matrix, epsilon,
                                                       gamma,
                                                      async_update, T,
                                                      Tmax, alpha, shared_array, step_queue, shared_array_base))

    producer2.start()

    producer1.join()
    producer2.join()

    step_list = []
    while not step_queue.empty():
        step_list.append(step_queue.get())

    return shared_array, step_list

# TODO: Need graphs as proof
# TODO: Pool instead of processes

if __name__ == '__main__':
    q_matrix, step_list = async_manager(processes=2,
                                        epsilon=0.55,
                                        alpha=0.45,
                                        gamma=0.95,
                                        async_update=5,
                                        Tmax=10000)

    print(q_matrix)
    print(step_list)