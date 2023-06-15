import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

warnings.filterwarnings("ignore")

from pyswarm import pso


# The objective function for optimization
def obj_fun(params):
    seera = pd.read_csv("datasets\SEERA_cleaned.csv", delimiter=',', decimal=".")
    feature_selection = True
    X = seera.drop('Effort', axis=1)
    y = seera['Effort']

    selected_features = ['Estimated  duration', 'Government policy impact',
                         'Developer incentives policy ', 'Developer training', 'Development team management',
                         'Top management opinion of previous system', 'User resistance',
                         ' Users stability ', ' Requirements flexibility ',
                         'Project manager experience', 'Precedentedness', 'Software tool experience', 'Team size',
                         'Team cohesion', 'Schedule quality', 'Development environment adequacy',
                         'Tool availability ', 'DBMS used', 'Technical stability', 'Degree of software reuse ',
                         ' Process reengineering ']

    if feature_selection:
        X_selected = X[selected_features]
    else:
        X_selected = X

    # Creazione dell'oggetto KFold con k=10
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)

    # Liste per memorizzare le performance dei modelli in ogni fold
    rmse_scores = []
    r2_scores = []
    mre_scores = []

    max_depth_rf = int(params[0])
    min_samples_split = float(params[1])
    n_estimators_rf = int(params[2])
    C = float(params[3])
    epsilon = float(params[4])
    gamma = float(params[5])
    kernel = ['linear', 'poly', 'rbf'][int(params[6])]
    n_neighbors = int(params[7])
    leaf_size = int(params[8])
    weights = ['uniform', 'distance'][int(params[9])]
    alpha = float(params[10])
    l1_ratio = float(params[11])
    learning_rate = float(params[12])
    n_estimators_gb = int(params[13])

    print('')
    print('Launched the iteration with ')
    print("max_depth_rf:", int(params[0]), "min_samples_split:", float(params[1]), "n_estimators_rf:", int(params[2]),
          "C:", float(params[3]), "epsilon:", float(params[4]), "gamma:", float(params[5]), "kernel:",
          ['linear', 'poly', 'rbf'][int(params[6])], "n_neighbors:", int(params[7]), "leaf_size:", int(params[8]),
          "weights:", ['uniform', 'distance'][int(params[9])], "alpha:", float(params[10]), "l1_ratio:",
          float(params[11]), "learning_rate:", float(params[12]), "n_estimators_gb:", int(params[13]))

    # Suddivisione dei dati utilizzando la k-fold cross-validation
    for train_indices, test_indices in kfold.split(X_selected):
        X_train = X_selected.iloc[train_indices]
        X_test = X_selected.iloc[test_indices]
        y_train = y.iloc[train_indices]
        y_test = y.iloc[test_indices]

        # Random Forest
        rf_regressor = RandomForestRegressor(
            n_estimators=n_estimators_rf,
            max_depth=max_depth_rf,
            min_samples_split=min_samples_split,
            random_state=42
        )
        rf_regressor.fit(X_train, y_train)

        # ElasticNet
        elasticnet_regressor = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio
        )
        elasticnet_regressor.fit(X_train, y_train)

        # SVR
        svr = SVR(
            C=C,
            epsilon=epsilon,
            gamma=gamma,
            kernel=kernel
        )
        svr.fit(X_train, y_train)

        # Gradient Boosting

        gb_regressor = GradientBoostingRegressor(
            learning_rate=learning_rate,
            random_state=42,
            n_estimators=n_estimators_gb
        )
        gb_regressor.fit(X_train, y_train)

        # KNN
        knn_regressor = KNeighborsRegressor(
            n_neighbors=n_neighbors,
            leaf_size=leaf_size,
            weights=weights
        )
        knn_regressor.fit(X_train, y_train)

        # Previsioni sul set di test
        y_pred_rf = rf_regressor.predict(X_test)
        y_pred_svr = svr.predict(X_test)
        y_pred_gb = gb_regressor.predict(X_test)
        y_pred_knn = knn_regressor.predict(X_test)
        y_pred_elastic = elasticnet_regressor.predict(X_test)

        X_meta = np.column_stack((y_pred_rf, y_pred_svr, y_pred_gb, y_pred_knn, y_pred_elastic))

        meta_regressor = LinearRegression()
        meta_regressor.fit(X_meta, y_test)

        # Previsioni sul set di test del meta-modello
        meta_pred = meta_regressor.predict(X_meta)

        # Calcolo delle misure di performance
        rmse = np.sqrt(mean_squared_error(y_test, meta_pred))
        r2 = r2_score(y_test, meta_pred)
        mre = np.mean(np.abs(y_test - meta_pred) / y_test) * 100

        # Aggiunta delle performance alle liste
        rmse_scores.append(rmse)
        r2_scores.append(r2)
        mre_scores.append(mre)

    # Calcolo delle medie delle performance
    mean_rmse = np.mean(rmse_scores)
    mean_r2 = np.mean(r2_scores)
    mean_mre = np.mean(mre_scores)

    normalized_rmse = mean_rmse / (mean_r2 + mean_mre)

    balanced_function = - mean_r2 + mean_mre + normalized_rmse

    return balanced_function


def pso_tuning():
    # Bounds for parameters space
    lb = [2, 0.1, 10, 0.01, 0.1, 0.01, 0, 1, 10, 0, 0, 0, 0.01, 10]
    ub = [64, 1.0, 1000, 1000, 2, 1.0, 2, 10, 100, 1, 1, 1, 0.1, 1000]

    # Run the optimization
    xopt, fopt = pso(obj_fun, lb, ub, swarmsize=20, maxiter=40, debug=True)
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
        'kernel': ['linear', 'poly', 'rbf'][int(xopt[6])],
        'n_neighbors': int(xopt[7]),
        'leaf_size': int(xopt[8]),
        'weights': ['uniform', 'distance'][int(xopt[9])],
        'alpha': float(xopt[10]),
        'l1_ratio': float(xopt[11]),
        'learning_rate': float(xopt[12]),
        'n_estimators_gb': int(xopt[13])
    }

    return algo_params