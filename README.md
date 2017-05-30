## Q Learning Implementations

1. [Q Learning](./src/qlearning.py)
2. [Asynchronous Q Learning](./src/async_qlearning,py)
3. [Pull Request on Asynchronous DQN (See Notes)](https://github.com/dbobrenko/async-deeprl/pull/2)


## Additional Work

[Hyperparamater Tuning](./src/hyperopt.py)

[Graphs for Q Learning and Async Q Learning](./src/qlearning.ipynb)

## Installation

With Anaconda: `conda env -f environment.yml`

## Run Tests

`pytest src/test`

## Notes
Part 2 was implemented using multiple processes.

I didn't have time to complete Part 3, so instead I talked to Dmitry Bobrenko, who wrote a multiprocess Python implementation of the Asynchronous DQN. I made some minor code improvements while I read his code.