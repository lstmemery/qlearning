## Q Learning Implementations

1. [Q Learning](./src/qlearning.py)
2. [Asynchronous Q Learning](./src/async_qlearning,py)
3. [Pull Request on Asynchronous DQN (See Notes)](https://github.com/dbobrenko/async-deeprl/pull/2)


## Additional Work

[Hyperparamater Tuning](./src/hyperopt.py)

[Graphs for Q Learning and Async Q Learning](./src/qlearning.ipynb)

## Installation

With Anaconda: `conda env -f environment.yml`

## How to Run

### Q Learning

`python src/q_learning.py`

```
Options:
  --episodes INTEGER   Number of Successful Runs.
  -e, --epsilon FLOAT  Probability of choosing the next action randomly (vs.
                       greedily).
  -a, --alpha FLOAT    Learning Rate.
  -g, --gamma FLOAT    Discount Factor.
  --help               Show this message and exit.
```

### Asynchronous Q Learning

`python src/async_qlearning.py`

```
Options:
  -n, --processes INTEGER  Number of Processes to run.
  -e, --epsilon FLOAT      Probability of choosing the next action randomly
                           (vs. greedily).
  -a, --alpha FLOAT        Learning Rate.
  -g, --gamma FLOAT        Discount Factor.
  -u, --update INTEGER     Number of steps until update.
  -s, --steps INTEGER      Total number of steps.
  --help                   Show this message and exit.
```

## Run Tests

`pytest src/test`

## Notes
Part 2 was implemented using multiple processes.

I didn't have time to complete Part 3, so instead I talked to Dmitry Bobrenko, who wrote a multiprocess Python implementation of the Asynchronous DQN. I made some minor code improvements while I read his code.