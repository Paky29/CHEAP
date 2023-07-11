import matplotlib.pyplot as plt
import optuna

from tuning.TuningModel import Ensemble


def objective(trial, clf_tune):

    if clf_tune=='rf':
      params = {
          'n_estimators' : trial.suggest_int("rf_n_estimators", 5, 100),
          'max_features' : trial.suggest_categorical('rf_max_features', ['auto', 'sqrt']),
          'max_depth' : trial.suggest_int("rf_max_depth", 10, 120),
          'min_samples_split' : trial.suggest_int('rf_min_samples_split', 2, 10),
          'min_samples_leaf' : trial.suggest_int('rf_min_samples_leaf', 1, 4)
      }

    elif clf_tune=='ada':
      params = {
          'n_estimators' : trial.suggest_int("n_estimators", 70, 90),
          'learning_rate' : trial.suggest_float("learning_rate", 0.01, 0.1),
          'loss' : trial.suggest_categorical('loss', ['linear', 'square', 'exponential'])
      }

    elif clf_tune=='svr':
      params = {
          'C' : trial.suggest_float("svr_C", 0.1, 5.0),
          'epsilon' : trial.suggest_float("svr_epsilon", 0.01, 0.5),
          'gamma' : trial.suggest_float("svr_gamma", 0.001, 0.1)
      }

    elif clf_tune=='en':
      params = {
          'alpha' : trial.suggest_float("en_alpha", 0.01, 1.0),
          'l1_ratio' : trial.suggest_float("en_l1_ratio", 0.1, 1.0),
          'tol' : trial.suggest_float("en_tol", 0.0001, 0.001)
      }

    else:
      params = {
          'n_neighbors' : trial.suggest_int("knn_n_neighbors", 5, 15),
          'leaf_size' : trial.suggest_int("knn_leaf_size", 1, 30),
          'weights' : trial.suggest_categorical("knn_weights", ['uniform', 'distance']),
          'p' : trial.suggest_int("knn_p", 1, 2)
      }


    ensemble = Ensemble()

    try:
      ensemble.load_data()
      mre = ensemble.StackingRegressor(params, clf_tune)
    except Exception as inst:
      print(inst)
      mre = 400

    return mre


def optuna_tuning():

    clf_tune='en'

    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, clf_tune), n_trials=20)

    print("BEST PARAMS")
    print(study.best_params)

    print("BEST VALUE")
    print(study.best_value)

    plot_mre = optuna.visualization.plot_param_importances(study)

    plot2_mre = optuna.visualization.plot_optimization_history(study)

    plot_mre.show()
    plot2_mre.show()
    plt.show()
