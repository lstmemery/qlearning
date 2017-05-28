from multiprocessing import Process, Queue, Value, Array, Manager
import numpy as np
import src.qlearning_numpy as ql
import time
from random import random
import ctypes

def qlearning_worker(r_matrix, epsilon, alpha, gamma, async_update, T, Tmax, q, global_q_matrix):
    # Initialize thread step count t <- 1 (Not sure why this starts at 1)
    t = 1
    delta_q_matrix = np.zeros_like(r_matrix).astype(float)

    start_state = ql.index_1d(5, 3)
    # Get initial state s
    state = start_state

    while T.value <= Tmax:
        # Choose action from state s using \epsilon-greedy policy
        with global_q_matrix.get_lock():
            np_q = to_numpy_array(global_q_matrix, r_matrix.shape)

        action = ql.get_epsilon_greedy_action(epsilon, np_q, state)
        # Take action a, observe r, s'
        reward = ql.peek_reward(r_matrix, state, action)
        next_state = ql.peek_next_state(r_matrix, state, action)
        updated_q = ql.update_q(delta_q_matrix, state, action, alpha, gamma, reward, next_state)
        # Accumulate updates:
        delta_q_matrix[state, action] = updated_q
        # s <- s'
        state = next_state
        # t <- t + 1 and T <- T + 1
        t += 1

        T.value += 1
        print(T.value)
        # if t % I_AsyncUpdate == 0 or s is terminal then
        if t % async_update == 0 or state != ql.index_1d(0, 8):
            # Perform async update
            send_local_q(q, delta_q_matrix)
            # clear updates \DeltaQ(s', a')
            delta_q_matrix = np.zeros_like(r_matrix).astype(float)
            if state != ql.index_1d(0, 8):
                state = start_state

    print("returning")


def receive_global_q(parent_conn, lock):
    try:
        lock.acquire()
        global_q = parent_conn.recv()
        return global_q
    finally:
        lock.release()


def send_local_q(q, delta_q_matrix):
    q.put(delta_q_matrix)


def acculmulate_q(q, global_q_matrix, T, Tmax):
    while T.value <= Tmax:
        if not q.empty():
            local_q_matrix = q.get()
            with global_q_matrix.get_lock():
                q_np = to_numpy_array(global_q_matrix, local_q_matrix.shape)
                q_np += local_q_matrix
                global_q_matrix = to_mp_array(q_np)

    print(q.empty())
    print("consumer returning")
    return


def to_mp_array(np_array):
    mp_array = Array(ctypes.c_double, np_array.flatten())
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

    global_q_matrix = to_mp_array(global_q_matrix_np)

    manager = Manager()
    q = manager.Queue()

    consumer = Process(target=acculmulate_q, args=(q, global_q_matrix,
                                                   T, Tmax))


    producer1 = Process(target=qlearning_worker, args=(r_matrix, epsilon,
                                                      alpha, gamma,
                                                      async_update, T,
                                                      Tmax, q, global_q_matrix))

    producer1.start()

    producer2 = Process(target=qlearning_worker, args=(r_matrix, epsilon,
                                                      alpha, gamma,
                                                      async_update, T,
                                                      Tmax, q, global_q_matrix))

    producer2.start()

    consumer.start()

    consumer.join()
    producer1.join()
    producer2.join()

    return to_numpy_array(global_q_matrix, r_matrix.shape)

if __name__ == '__main__':
    print(async_manager(processes=2,
                  epsilon=0.1,
                  alpha=0.1,
                  gamma=0.95,
                  async_update=4,
                  Tmax=100))