from multiprocessing import Process, Queue, Pipe, Value, Lock
import numpy as np
import src.qlearning_numpy as ql
import time
from random import random

def qlearning_worker(r_matrix, epsilon, alpha, gamma, async_update, T, Tmax, q, parent_conn, lock):
    # Initialize thread step count t <- 1 (Not sure why this starts at 1)
    t = 1
    delta_q_matrix = np.zeros_like(r_matrix).astype(float)

    start_state = ql.index_1d(5, 3)
    # Get initial state s
    state = start_state

    while T.value <= Tmax:
        # Choose action from state s using \epsilon-greedy policy
        global_q_matrix = None

        while not global_q_matrix:
            global_q_matrix = receive_global_q(parent_conn, lock)
            if not global_q_matrix:
                time.sleep(random() * 0.1)

        action = ql.get_epsilon_greedy_action(epsilon, global_q_matrix, state)
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
        # if t % I_AsyncUpdate == 0 or s is terminal then
        if t % async_update == 0 or state != ql.index_1d(0, 8):
            # Perform async update
            send_local_q(q, delta_q_matrix)
            # clear updates \DeltaQ(s', a')
            delta_q_matrix = np.zeros_like(r_matrix).astype(float)
            if state != ql.index_1d(0, 8):
                state = start_state


def receive_global_q(parent_conn, lock):
    try:
        lock.acquire()
        return parent_conn.recv()
    finally:
        lock.release()


def send_local_q(q, delta_q_matrix):
    q.put(delta_q_matrix)


def acculmulate_q(q, parent_conn, child_conn, T, Tmax, lock):
    while T.value <= Tmax:
        if not q.empty():
            global_q_matrix = None

            while not global_q_matrix:
                global_q_matrix = receive_global_q(parent_conn, lock)
                if not global_q_matrix:
                    time.sleep(random() * 0.1)

            local_q_matrix = q.get()
            global_q_matrix += local_q_matrix
            child_conn.send(global_q_matrix)

        time.sleep(random() * 0.1)


def async_manager(threads, epsilon, alpha, gamma, async_update, Tmax):
    # Assume global shared Q(s, a) function values, and counter T = 0
    T = Value('i', 0)
    r_matrix = ql.make_transition_matrix(ql.state_grid)
    # Initialize global Q(s, a)
    global_q_matrix = np.zeros_like(r_matrix).astype(float)

    q = Queue()

    parent_conn, child_conn = Pipe()
    lock = Lock()

    parent_conn.send(global_q_matrix)


    producer1 = Process(target=qlearning_worker, args=(r_matrix, epsilon,
                                                      alpha, gamma,
                                                      async_update, T,
                                                      Tmax, q, parent_conn,
                                                      lock))
    producer1.start()

    producer2 = Process(target=qlearning_worker, args=(r_matrix, epsilon,
                                                      alpha, gamma,
                                                      async_update, T,
                                                      Tmax, q, parent_conn,
                                                      lock))
    producer2.start()

    consumer = Process(target=acculmulate_q, args=(q, parent_conn,
                                                   child_conn, T, Tmax, lock))
    consumer.start()

    while T.value <= Tmax:
        time.sleep(random())

    global_q_matrix = parent_conn.recv()

    producer1.join()
    producer2.join()
    consumer.join()

    return global_q_matrix




if __name__ == '__main__':
    print(async_manager(threads=2,
                  epsilon=0.1,
                  alpha=0.1,
                  gamma=0.95,
                  async_update=4,
                  Tmax=100))