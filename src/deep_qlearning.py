import keras as kr
import gym
from keras.models import Sequential
from keras.layers import Convolution2D, Dense, Activation


def build_model():
    model = Sequential()
    model.add(Convolution2D(filters=16, kernel_size=(8, 8), strides=4))
    model.add(Activation('relu'))
    model.add(Convolution2D(filters=32, kernel_size=(4, 4), strides=2))
    model.add(Activation('relu'))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('linear'))

if __name__ == '__main__':
    space_invaders = gym.make("SpaceInvaders-v0")
