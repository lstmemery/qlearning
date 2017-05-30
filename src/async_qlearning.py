# coding=utf-8
from multiprocessing import Process, Value, Array, Manager, Lock, Pool
import numpy as np
import src.qlearning as ql
import ctypes
from random import random, randint

updated_matrix = ql.make_transition_matrix(ql.updated_grid)


def update_delta_q(local_q_matrix, state, action, gamma, reward, next_state, global_q_state):
    """Update a local Q matrix, based on the the state and next state of the global Q matrix.

    ∆Q(s, a) ← ∆Q(s, a) + r + γ max a Q(s 0 , a) − Q(s, a)

    Parameters
    ----------
    local_q_matrix : ndararry of floats
        An 2D Q matrix representing local updates to the Q function.
    state : int
        The current state, represented as a row number in the transition matrix.
    action : int
        The next action to be taken, represented as a column number in the transition matrix.
    gamma : float
        The discount factor. The weight that the algorithm will consider future rewards. gamma = 0 is entirely greedy.
        Values higher than 1 can cause the algorithm to diverge.[1]
    reward : int
        The reward in the transition matrix.
    next_state : int
        The next state of the transition matrix.
    global_q_state : ndarray of floats
        An 2D Q matrix, representing the global Q function. This 2D matrix is not updated, so it does not require it's
        lock.

    Returns
    -------
    next_q : float
        The value the Q matrix will be updated with.

    References
    ----------
    1. Q-learning. In: Wikipedia [Internet]. 2017 [cited 2017 May 29]. Available from: https://en.wikipedia.org/w/index.php?title=Q-learning&oldid=762556833

    Notes
    -----
    Unlike the synchronous version, this version does not require a. Since a is the same for all local Q updates,
     it can be applied when the change to the local Q matrix are applied to the global Q matrix.
    """
    max_next_step = max(global_q_state[next_state, :])
    next_q = local_q_matrix[state, action] + (reward + (gamma * max_next_step) - global_q_state[state, action])
    return next_q


def qlearning_worker(r_matrix, epsilon, gamma, async_update, T, Tmax, alpha, global_q_matrix, step_queue, raw_array):
    # Initialize thread step count t <- 1 (Not sure why this starts at 1)
    t = 1
    last_reward = 1
    total_reward = 0
    delta_q_matrix = np.zeros_like(r_matrix).astype(float)

    start_state = ql.index_1d(5, 3)
    final_state = ql.index_1d(0, 8)

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
        if t % async_update == 0 or state == final_state:
            # Perform async update
            with raw_array.get_lock():
                global_q_matrix[delta_q_matrix.nonzero()] += alpha * delta_q_matrix[delta_q_matrix.nonzero()]

            # clear updates \DeltaQ(s', a')
            delta_q_matrix = np.zeros_like(r_matrix).astype(float)
            if state == final_state:
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

    workers = []
    for _ in range(processes):
        worker = Process(target=qlearning_worker, args=(r_matrix, epsilon, gamma,
                                                      async_update, T,
                                                      Tmax, alpha, shared_array, step_queue, shared_array_base))
        workers.append(worker)
        worker.start()

    for worker in workers:
        worker.join()

    step_list = []

    while not step_queue.empty():
        step_list.append(step_queue.get())

    return shared_array, step_list

# TODO: Need graphs as proof

if __name__ == '__main__':
    q_matrix, step_list = async_manager(processes=6,
                                        epsilon=0.55,
                                        alpha=0.45,
                                        gamma=0.95,
                                        async_update=5,
                                        Tmax=50000)

    print(q_matrix)
    print(step_list)