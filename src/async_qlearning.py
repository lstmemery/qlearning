from multiprocessing import Process, Queue, Pipe
import numpy as np
import src.qlearning_numpy as ql


def threaded_qlearning(r_matrix, state, epsilon, alpha, gamma, async_update, Tmax, q):
    # Initialize thread step count t <- 1
    t = 1
    global global_q_matrix
    delta_q_matrix = np.zeros_like(r_matrix).astype(float)
    global T

    while T <= Tmax:
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
        T += 1
        if t % async_update == 0 or state != ql.index_1d(0, 8):
            send_local_q(q, delta_q_matrix)
            delta_q_matrix = np.zeros_like(r_matrix).astype(float)

def send_local_q(q, delta_q_matrix):
    q.put(delta_q_matrix)


def acculmulate_q(q, global_q_matrix, child_conn):
    local_q_matrix = q.get()
    global_q_matrix += local_q_matrix
    child_conn.send(global_q_matrix)

def async_qlearning(grid, epsilon, async_update, Tmax):
    process_queue = Queue(10)
    global T
    while T < Tmax:
        # Assume global shared Q(s, a) function values, and counter T = 0
        r_matrix = ql.make_transition_matrix(grid)
        global_q_matrix = np.zeros_like(r_matrix).astype(float)

        # Initialize global Q(s, a)
    # Initialize thread updates \Delta Q(s, a)
        pool = Process()

    # Get initial state s
    # repeat



        # s <- s'
        # t <- t + 1 and T <- T + 1
        # if t % I_AsyncUpdate == 0 or s is terminal then
            # Perform async update
            # clear updates \DeltaQ(s', a')
    # until T > T_max
    return global_q_matrix

if __name__ == '__main__':
    T = 0
    r_matrix = ql.make_transition_matrix(ql.state_grid)
    global_q_matrix = np.zeros_like(r_matrix).astype(float)