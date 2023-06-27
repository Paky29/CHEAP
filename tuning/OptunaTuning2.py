import numpy as np
import optuna
import pandas as pd
import sklearn
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVR


class Ensemble:
    def __init__(self):
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.k = 10

    def load_data(self):
        seera = pd.read_csv("../datasets/SEERA_train.csv", delimiter=',', decimal=".")
        X = seera.drop(['Indice Progetto', 'Actual effort'], axis=1)
        y = seera['Actual effort']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=23)

    def StackingClassifier(self, params):

        max_depth_rf = int(params[0])
        min_samples_split = float(params[1])
        n_estimators_rf = int(params[2])
        C = float(params[3])
        epsilon = float(params[4])
        gamma = float(params[5])
        # kernel = ['poly', 'rbf', 'sigmoid'][int(x[6])]
        kernel = 'sigmoid'
        n_neighbors = int(params[7])
        leaf_size = int(params[8])
        weights = params[9]
        alpha = float(params[10])
        l1_ratio = float(params[11])
        learning_rate = float(params[12])
        n_estimators_gb = int(params[13])

        hidden_layer_sizes = int(params[14])
        momentum = float(params[15])
        activation = params[16]
        alpha_mlp = float(params[17])


        rf_regressor = RandomForestRegressor(max_depth=max_depth_rf, min_samples_split=min_samples_split, n_estimators=n_estimators_rf)
        svr = SVR(C=C, epsilon=epsilon, gamma=gamma, kernel=kernel)

        elasticnet_regressor = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)

        knn_regressor = KNeighborsRegressor(n_neighbors=n_neighbors, leaf_size=leaf_size, weights=weights)

        gb_regressor = GradientBoostingRegressor(learning_rate=learning_rate, n_estimators=n_estimators_gb)

        # Define weak learners
        weak_learners = [('rf', rf_regressor),
                         ('knn', knn_regressor),
                         ('svr', svr),
                         ('gb', gb_regressor),
                         ('en', elasticnet_regressor)]

        # Final learner or meta model
        final_learner = MLPRegressor(random_state=1, max_iter=500, hidden_layer_sizes=hidden_layer_sizes, momentum=momentum, activation='identity', solver = 'lbfgs', alpha=alpha_mlp)

        train_meta_model = None
        test_meta_model = None

        # Start stacking
        for clf_id, clf in weak_learners:
            # Predictions for each classifier based on k-fold
            predictions_clf = self.k_fold_cross_validation(clf)

            # Predictions for test set for each classifier based on train of level 0
            test_predictions_clf = self.train_level_0(clf)

            # Stack predictions which will form
            # the input data for the data model
            if isinstance(train_meta_model, np.ndarray):
                train_meta_model = np.vstack((train_meta_model, predictions_clf))
            else:
                train_meta_model = predictions_clf

            # Stack predictions from test set
            # which will form test data for meta model
            if isinstance(test_meta_model, np.ndarray):
                test_meta_model = np.vstack((test_meta_model, test_predictions_clf))
            else:
                test_meta_model = test_predictions_clf

        # Transpose train_meta_model
        train_meta_model = train_meta_model.T

        # Transpose test_meta_model
        test_meta_model = test_meta_model.T

        # Training level 1
        return self.train_level_1(final_learner, train_meta_model, test_meta_model)

    def k_fold_cross_validation(self, clf):

        predictions_clf = None

        # Number of samples per fold
        batch_size = int(len(self.X_train) / self.k)

        # Stars k-fold cross validation
        for fold in range(self.k):

            # Settings for each batch_size
            if fold == (self.k - 1):
                test = self.X_train.iloc[(batch_size * fold):, :]
                batch_start = batch_size * fold
                batch_finish = self.X_train.shape[0]
            else:
                test = self.X_train.iloc[(batch_size * fold): (batch_size * (fold + 1)), :]
                batch_start = batch_size * fold
                batch_finish = batch_size * (fold + 1)

            # test & training samples for each fold iteration
            fold_X_test = self.X_train.iloc[batch_start:batch_finish, :]
            fold_x_train = self.X_train.iloc[[index for index in range(self.X_train.shape[0]) if
                                              index not in range(batch_start, batch_finish)], :]

            # test & training targets for each fold iteration
            fold_y_test = self.y_train.iloc[batch_start:batch_finish]
            fold_y_train = self.y_train.iloc[
                [index for index in range(self.X_train.shape[0]) if index not in range(batch_start, batch_finish)]]

            # Fit current classifier
            clf.fit(fold_x_train, fold_y_train)
            fold_y_pred = clf.predict(fold_X_test)

            # Store predictions for each fold_X_test
            if isinstance(predictions_clf, np.ndarray):
                predictions_clf = np.concatenate((predictions_clf, fold_y_pred))
            else:
                predictions_clf = fold_y_pred

        return predictions_clf

    def train_level_0(self, clf):
        # Train in full real training set
        clf.fit(self.X_train, self.y_train)
        # Get predictions from full real test set
        y_pred = clf.predict(self.X_test)

        return y_pred

    def train_level_1(self, final_learner, train_meta_model, test_meta_model):
        # Train is carried out with final learner or meta model
        final_learner.fit(train_meta_model, self.y_train)
        # Getting train and test accuracies from meta_model

        predictions = final_learner.predict(test_meta_model)

        # Calcolo delle misure di performance
        rmse = np.sqrt(mean_squared_error(self.y_test, predictions))
        r2 = r2_score(self.y_test, predictions)
        mre = np.mean(np.abs(self.y_test - predictions) / self.y_test) * 100

        print(f"Test rmse: {rmse}")
        print(f"Test mre: {mre}")
        print(f"Train r2: {final_learner.score(train_meta_model, self.y_train)}")
        print(f"Test r2: {final_learner.score(test_meta_model, self.y_test)}")

        return mre, r2, rmse


def objective(trial):

    params = []

    max_depth_rf = trial.suggest_int("rf_max_depth", 2, 10)
    min_samples_split = trial.suggest_float("rf_min_samples_split", 0.01, 0.5)
    n_estimators_rf = trial.suggest_int("rf_n_estimators", 100, 1000)

    params.append(max_depth_rf)
    params.append(min_samples_split)
    params.append(n_estimators_rf)

    # SVR
    C = trial.suggest_float("svr_C", 0.1, 5.0)
    epsilon = trial.suggest_float("svr_epsilon", 0.01, 0.5)
    gamma = trial.suggest_float("svr_gamma", 0.001, 0.1)
    kernel = trial.suggest_categorical("svr_kernel", ['linear', 'rbf', 'sigmoid'])

    params.append(C)
    params.append(epsilon)
    params.append(gamma)
    params.append(kernel)

    # KNN
    n_neighbors = trial.suggest_int("knn_n_neighbors", 3, 10)
    leaf_size = trial.suggest_int("knn_leaf_size", 10, 50)
    weights = trial.suggest_categorical("knn_weights", ['uniform', 'distance'])

    params.append(n_neighbors)
    params.append(leaf_size)
    params.append(weights)

    # ElasticNet
    alpha = trial.suggest_float("en_alpha", 0.01, 1.0)
    l1_ratio = trial.suggest_float("en_l1_ratio", 0.1, 1.0)

    params.append(alpha)
    params.append(l1_ratio)

    # Gradient Boosting
    learning_rate = trial.suggest_float("gb_learning_rate", 0.01, 0.1)
    n_estimators_gb = trial.suggest_int("gb_n_estimators", 100, 1000)

    params.append(learning_rate)
    params.append(n_estimators_gb)

    hidden_layer_sizes = trial.suggest_int("hidden_layer_sizes", 10, 100)
    momentum = trial.suggest_float("momentum", 0.1, 0.9)
    activation = trial.suggest_categorical("activation", ['identity', 'relu'])
    alpha_mlp = trial.suggest_float("alpha", 0.0001, 0.1)


    params.append(hidden_layer_sizes)
    params.append(momentum)
    params.append(activation)
    params.append(alpha_mlp)


    ensemble = Ensemble()
    ensemble.load_data()
    return ensemble.StackingClassifier(params)


def optuna_tuning():
    sampler = optuna.samplers.NSGAIISampler(population_size=100)
    study = optuna.create_study(sampler=sampler, directions=["minimize", "maximize", "minimize"])
    study.optimize(objective, n_trials=20)
    trials = study.best_trials
    best_params = []
    for trial in trials:
        print("Best Score: ", trial.values)
        print("Best Params: ")
        for key, value in trial.params.items():
            print("  {}: {}".format(key, value))




    '''plot_mre = optuna.visualization.plot_param_importances(study, target=lambda t: t.values[0], target_name="mean_mre")

    plot2_mre = optuna.visualization.plot_optimization_history(study, target=lambda t: t.values[0],
                                                               target_name="mean_mre")

    plot_r2 = optuna.visualization.plot_param_importances(study, target=lambda t: t.values[1], target_name="mean_r2")

    plot2_r2 = optuna.visualization.plot_optimization_history(study, target=lambda t: t.values[1],
                                                              target_name="mean_r2")

    plot_rmse = optuna.visualization.plot_param_importances(study, target=lambda t: t.values[2],
                                                            target_name="mean_rmse")

    plot2_rmse = optuna.visualization.plot_optimization_history(study, target=lambda t: t.values[2],
                                                                target_name="mean_rmse")

    plot_mre.show()
    plot2_mre.show()
    plot_r2.show()
    plot2_r2.show()
    plot_rmse.show()
    plot2_rmse.show()
    plt.show()'''


def scaling_robust(seera):
    isEffort = False
    if 'Actual effort' in seera.columns:
        effort = seera['Actual effort']
        seera = seera.drop(['Actual effort'], axis=1)
        isEffort = True

    isIndex = False
    if 'Indice Progetto' in seera.columns:
        indici = seera['Indice Progetto']
        seera = seera.drop('Indice Progetto', axis=1)
        isIndex = True

    x_columns = seera.columns
    # Create an instance of the RobustScaler
    scaler = RobustScaler()

    # Fit the scaler to the data and transform it
    scaled_data = scaler.fit_transform(seera)

    seera = pd.DataFrame(scaled_data, columns=x_columns)
    if isEffort:
        seera['Actual effort'] = effort
    if isIndex:
        seera.insert(0, 'Indice Progetto', indici)
    return seera


if __name__ == "__main__":
    optuna_tuning()
