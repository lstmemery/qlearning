# coding=utf-8
# Implement Q-learning algorithm 1 for solving the Gridworld (Section 4).
# We recommend implementing a Q(s, a) function as a simple 2-dimensional
# array, since there are only 46 states and 4 actions from each state. Initialize
# Q(s, a) values with 0 for all pairs (s, a). Set discount factor γ = 0.95.
# Choose the values of all other necessary parameters as you see fit, a little
# exploration of a parameters space can go a long way.

from copy import deepcopy
from random import choice, random

from gridworld import GridWorld

def q_learning(grid, episode_limit, epsilon):
    action_grid = [[0 for _ in range(grid.x)] for _ in range(grid.y)]
    options = ("up", "down", "left", "right")

    for episode in range(episode_limit):
        test_grid = deepcopy(grid)
        while test_grid.agent.reward == 0:
            if epsilon < random:
                action = choice(options)
            else:
                pass

def get_best_neighbour(state_grid, action_grid):
    neighbours = state_grid.agent.get_neighbours()

    best_neighbour = (None, 0)
    for neighbour in neighbours:
        if action_grid[neighbour[1]][neighbour[2]] > best_neighbour[1]:
            best_neighbour = (neighbour[0], action_grid[neighbour[1]][neighbour[2]])

    return best_neighbour[0]



