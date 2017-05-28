from hyperopt import hp, fmin, tpe
import qlearning as ql



def objective(params):
    q, iterations = ql.qlearning(ql.state_grid,
                                 100,
                                 epsilon=params['epsilon'],
                                 alpha=params['alpha'],
                                 gamma=0.95,
                                 updated_grid=ql.updated_grid)
    return sum(iterations[-5:])/5


if __name__ == '__main__':
    space = {'alpha': hp.quniform('alpha', 0.05, 1, 0.05),
             'epsilon': hp.quniform('epsilon', 0.05, 1, 0.05)}

    best = fmin(objective, space=space, algo=tpe.suggest, max_evals=100)
    print(best)

    # Best Overall
    # {'alpha': 0.45, 'epsilon': 0.55}

    # Best of the last 5 rounds
    # {'alpha': 0.5, 'epsilon': 0.05}