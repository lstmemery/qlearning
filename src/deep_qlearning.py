from collections import deque

import gym
from keras.models import Sequential
from keras.layers import Convolution2D, Dense, Activation
import numpy as np
import random
import tensorflow as tf

# I worked through this tutorial https://keon.io/deep-q-learning/
# That tutorial is not asynchronous, though
# This code closely resembles what I'm attempting


class DQNAgent(object):

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.90
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.frames = 5
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Convolution2D(filters=16, kernel_size=(8, 8), strides=4, input_shape=(84, 84, self.frames)))
        model.add(Activation('relu'))
        model.add(Convolution2D(filters=32, kernel_size=(4, 4), strides=2))
        model.add(Activation('relu'))
        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(Dense(1))
        model.add(Activation('linear'))
        model.compile(loss='mse', optimizer='rmsprop')

        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action


    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
              target = reward + self.gamma * \
                       np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

if __name__ == '__main__':
    episodes = 10
    done = False

    space_invaders = gym.make("SpaceInvaders-v0")

    state_size = space_invaders.observation_space.shape[0]
    action_size = space_invaders.action_space.n

    agent = DQNAgent(state_size, action_size)

    for e in range(episodes):

        # reset state in the beginning of each game
        state = space_invaders.reset()
        state = np.reshape(state, [1, 4])

        # time_t represents each frame of the game
        # the more time_t the more score
        for time_t in range(500):
            # turn this on if you want to render
            space_invaders.render()

            # Decide action
            action = agent.act(state)

            # Advance the game to the next frame based on the action.
            # Reward is 1 for every frame the pole survived
            next_state, reward, done, _ = space_invaders.step(action)
            next_state = np.reshape(next_state, [1, 4])

            # Remember the previous state, action, reward, and done
            agent.remember(state, action, reward, next_state, done)

            # make next_state the new current state for the next frame.
            state = next_state

            # done becomes True when the game ends
            # ex) The agent drops the pole
            if done:
                # print the score and break out of the loop
                print("episode: {}/{}, score: {}"
                      .format(e, episodes, time_t))
                break
        # train the agent with the experience of the episode
        agent.replay(32)
