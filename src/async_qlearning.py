import threading
import numpy as np
import src.qlearning_numpy as ql

from Queue import Queue


def threaded_qlearning(r_matrix, state, epsilon, alpha, gamma, async_update):
    t = 1
    global global_q_matrix
    delta_q_matrix = np.zeros_like(r_matrix).astype(float)
    while t % async_update == 0 and state != ql.index_1d(0, 8):

        # Choose action from state s using \epsilon-greedy policy
        action = ql.get_epsilon_greedy_action(epsilon, global_q_matrix, state)
        # Take action a, observe r, s'
        reward = ql.peek_reward(r_matrix, state, action)
        next_state = ql.peek_next_state(r_matrix, state, action)
        updated_q = ql.update_q(delta_q_matrix, state, action, alpha, gamma, reward, next_state)
        # Accumulate updates:
        delta_q_matrix[state, action] = updated_q
        state = next_state
        t += 1

    return delta_q_matrix


def async_qlearning(grid, epsilon, async_update, Tmax):
    thread_queue = Queue(10)
    T = 0
    global T
    while T < Tmax:
        # Assume global shared Q(s, a) function values, and counter T = 0
        r_matrix = ql.make_transition_matrix(grid)
        global_q_matrix = np.zeros_like(r_matrix).astype(float)
        # Initialize thread step count t <- 1
        t = 1
        # Initialize global Q(s, a)
    # Initialize thread updates \Delta Q(s, a)
        threading.Thread(target=threaded_qlearning).start()
    # Get initial state s
    # repeat



        # s <- s'
        # t <- t + 1 and T <- T + 1
        # if t % I_AsyncUpdate == 0 or s is terminal then
            # Perform async update
            # clear updates \DeltaQ(s', a')
    # until T > T_max
    return global_q_matrix