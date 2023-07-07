import warnings
from pyswarm import pso

from tuning.TuningModel import Ensemble

warnings.filterwarnings("ignore")


# The objective function for optimization
def objective(x, *args):
    clf_tune, = args

    max_depth_rf = int(x[0])
    min_samples_split = float(x[1])
    n_estimators_rf = int(x[2])
    C = float(x[3])
    epsilon = float(x[4])
    gamma = float(x[5])
    kernel = 'sigmoid'
    n_neighbors = int(x[6])
    leaf_size = int(x[7])
    weights = ['uniform', 'distance'][int(x[8])]
    alpha = float(x[9])
    l1_ratio = float(x[10])
    learning_rate = float(x[11])
    n_estimators_ada = int(x[12])

    if clf_tune == 'rf':
        params = {
            'n_estimators': n_estimators_rf,
            'max_depth': max_depth_rf,
            'min_samples_split': min_samples_split
        }

    elif clf_tune == 'ada':
        params = {
            'n_estimators': n_estimators_ada,
            'learning_rate': learning_rate
        }

    elif clf_tune == 'svr':
        params = {
            'C': C,
            'epsilon': epsilon,
            'gamma': gamma
        }

    elif clf_tune == 'en':
        params = {
            'alpha': alpha,
            'l1_ratio': l1_ratio
        }

    else:
        params = {
            'n_neighbors': n_neighbors,
            'leaf_size': leaf_size,
            'weights': weights
        }

    ensemble = Ensemble()

    try:
        ensemble.load_data()
        mre = ensemble.BlendingRegressor(params, clf_tune)
    except Exception as inst:
        mre = 10000

    return mre


def pso_tuning():
    # Bounds for parameters space
    lb = [2, 0.1, 10, 0.01, 0.01, 0.001, 3, 10, 0, 0, 0, 0.01, 10]
    ub = [32, 0.3, 400, 600, 1.1, 0.04, 10, 55, 1, 1, 1, 0.05, 500]

    # Run the optimization
    xopt, fopt = pso(objective, lb, ub, swarmsize=20, maxiter=8, debug=True, args=('knn',))
    print('OPTIMAL PARAMETERS:')
    print(xopt, fopt)

    # Store the best params
    algo_params = {
        'max_depth_rf': int(xopt[0]),
        'min_samples_split': float(xopt[1]),
        'n_estimators_rf': int(xopt[2]),
        'C': float(xopt[3]),
        'epsilon': float(xopt[4]),
        'gamma': float(xopt[5]),
        'n_neighbors': int(xopt[6]),
        'leaf_size': int(xopt[7]),
        'weights': ['uniform', 'distance'][int(xopt[8])],
        'alpha': float(xopt[9]),
        'l1_ratio': float(xopt[10]),
        'learning_rate': float(xopt[11]),
        'n_estimators_ada': int(xopt[12])
    }

    print(algo_params)
