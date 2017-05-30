# coding=utf-8
from multiprocessing import Process, Value, Array, Manager, Lock
import numpy as np
import src.qlearning as ql
import ctypes
import time
from random import random

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


def qlearning_worker(r_matrix, epsilon, gamma, async_update, T, Tmax, q, global_q_matrix, step_queue):
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
        # while not q.empty():
        #     time.sleep(0.01)
        # with global_q_matrix.get_lock():
        #     np_q = to_numpy_array(global_q_matrix, r_matrix.shape)

        print("Selecting")
        action = ql.get_epsilon_greedy_action(epsilon, global_q_matrix, state)
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
            global_q_matrix[delta_q_matrix.nonzero()] += 0.5 * delta_q_matrix[delta_q_matrix.nonzero()]
            # send_local_q(q, delta_q_matrix)

            # print("put")
            # clear updates \DeltaQ(s', a')
            delta_q_matrix = np.zeros_like(r_matrix).astype(float)
            if state == ql.index_1d(0, 8):
                step_queue.put(t - last_reward)
                last_reward = t
                state = start_state
                total_reward += 1


    # print(total_reward)
    # print(t / total_reward)


def send_local_q(q, delta_q_matrix):
    q.put(delta_q_matrix)


def acculmulate_q(q, global_q_matrix, T, Tmax, alpha, lock):
    q_np = None
    while T.value <= Tmax or not q.empty():
        if not q.empty():
            local_q_matrix = q.get()
            print("get")
            with global_q_matrix.get_lock():
                q_np = to_numpy_array(global_q_matrix, local_q_matrix.shape)
                q_np[local_q_matrix.nonzero()] += alpha * local_q_matrix[local_q_matrix.nonzero()]
                global_q_matrix = to_mp_array(q_np, lock)

    q.put(q_np)
    return


def to_mp_array(np_array, lock):
    mp_array = Array(ctypes.c_double, np_array.flatten(), lock=lock)
    return mp_array


def to_numpy_array(mp_array, dim):
    np_array = np.frombuffer(mp_array.get_obj())
    return np_array.reshape(*dim)


def async_manager(processes, epsilon, alpha, gamma, async_update, Tmax):
    # Assume global shared Q(s, a) function values, and counter T = 0
    T = Value('i', 0, lock=False)
    r_matrix = ql.make_transition_matrix(ql.state_grid)
    # Initialize global Q(s, a)

    global_q_matrix_np = np.zeros_like(r_matrix).astype(float)
    manager = Manager()

    q = manager.Queue()
    lock = Lock()

    shared_array_base = Array(ctypes.c_double, r_matrix.shape[0] * r_matrix.shape[1])
    shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
    shared_array = shared_array.reshape(r_matrix.shape)

    global_q_matrix = to_mp_array(global_q_matrix_np, lock=lock)
    step_queue = manager.Queue(2048)

    # consumer = Process(target=acculmulate_q, args=(q, shared_array,
    #                                                T, Tmax, alpha, lock))

    producer1 = Process(target=qlearning_worker, args=(r_matrix, epsilon, gamma,
                                                      async_update, T,
                                                      Tmax, q, shared_array, step_queue))

    producer1.start()

    producer2 = Process(target=qlearning_worker, args=(r_matrix, epsilon,
                                                       gamma,
                                                      async_update, T,
                                                      Tmax, q, shared_array, step_queue))

    producer2.start()

    # consumer.start()
    #
    # consumer.join()
    producer1.join()
    producer2.join()

    # q_np = q.get()

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