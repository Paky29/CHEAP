import numpy as np
import optuna
import pandas as pd
import sklearn
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR


def objective(trial):
    seera = pd.read_csv("datasets/SEERA_imputated.csv", delimiter=',', decimal=".")
    X = seera.drop('Effort', axis=1)
    y = seera['Effort']

    selected_features = ['Estimated  duration', 'Government policy impact',
                         'Developer incentives policy ', 'Developer training', 'Development team management',
                         'Top management opinion of previous system', 'User resistance',
                         ' Users stability ', ' Requirements flexibility ',
                         'Project manager experience', 'Precedentedness', 'Software tool experience', 'Team size',
                         'Team cohesion', 'Schedule quality', 'Development environment adequacy',
                         'Tool availability ', 'DBMS used', 'Technical stability', 'Degree of software reuse ', ' Process reengineering ',
                         'Technical documentation']

    X_selected = X[selected_features]

    # Suddivisione dei dati utilizzando la k-fold cross-validation
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.30, random_state=42)

    max_depth_rf = trial.suggest_int("rf_max_depth", 2, 64, log=True)
    min_samples_split = trial.suggest_float("rf_min_sample_split", 0, 1)
    n_estimators_rf = trial.suggest_int("rf_n_estimators", 10, 1000)

    C = trial.suggest_float("svr_C", 0.01, 1000)
    epsilon = trial.suggest_float("svr_epsilon", 0.1, 2)
    gamma = trial.suggest_float("svr_gamma", 0.01, 1.0)
    kernel = trial.suggest_categorical("svr_kernel", ['linear', 'poly', 'rbf'])

    n_neighbors = trial.suggest_int("knn_n_neighbors", 1, 10)
    leaf_size = trial.suggest_int("knn_leaf_size", 10, 100)
    weights = trial.suggest_categorical("knn_weights", ['uniform', 'distance'])

    alpha = trial.suggest_float("en_alpha", 0, 1)
    l1_ratio = trial.suggest_float("en_l1_ratio", 0, 1)

    learning_rate = trial.suggest_float("gb_learning_rate", 0.01, 0.1)
    n_estimators_gb = trial.suggest_int("gb_n_estimators", 10, 1000)

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
        kernel= kernel
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
        weights= weights
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

    if mre<72:
        print('RMSE:', rmse)
        print('R^2 score:', r2)
        print('MRE:', mre)
        print('-------------------------')

    return mre


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=150)
    trial = study.best_trial
    print("Best Score: ", trial.value)
    print("Best Params: ")
    for key, value in trial.params.items():
        print("  {}: {}".format(key, value))

    plot = optuna.visualization.plot_param_importances(study)

    plot2 = optuna.visualization.plot_optimization_history(study)

    plot.show()
    plot2.show()
    plt.show()
