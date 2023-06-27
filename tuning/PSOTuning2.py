import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

warnings.filterwarnings("ignore")

from pyswarm import pso

class Ensemble:
    def __init__(self):
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.k = 10

    def load_data(self):
        seera = pd.read_csv("../datasets/SEERA_train.csv", delimiter=',', decimal=".")
        #X = seera[['Customer organization type', 'Estimated  duration', 'Organization management structure clarity', 'Developer hiring policy', 'Project manager experience', 'Team size', 'Degree of risk management', 'Required reusability']]
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
        weights = ['uniform', 'distance'][int(params[9])]
        alpha = float(params[10])
        l1_ratio = float(params[11])
        learning_rate = float(params[12])
        n_estimators_gb = int(params[13])

        hidden_layer_sizes = int(params[14])
        momentum = float(params[15])
        activation = ['identity', 'relu'][int(params[16])]
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
        #ACTIVATION E SOLVER FISSATI
        final_learner = MLPRegressor(random_state=1, max_iter=500, hidden_layer_sizes=hidden_layer_sizes, momentum=momentum, activation='identity', solver='lbfgs', alpha=alpha_mlp)

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

# The objective function for optimization
def obj_fun(x):
    print('')
    print('Launched the iteration with ' + str(x))

    max_depth_rf = int(x[0])
    min_samples_split = float(x[1])
    n_estimators_rf = int(x[2])
    C = float(x[3])
    epsilon = float(x[4])
    gamma = float(x[5])
    # kernel = ['poly', 'rbf', 'sigmoid'][int(x[6])]
    kernel = 'sigmoid'
    n_neighbors = int(x[7])
    leaf_size = int(x[8])
    weights = ['uniform', 'distance'][int(x[9])]
    alpha = float(x[10])
    l1_ratio = float(x[11])
    learning_rate = float(x[12])
    n_estimators_gb = int(x[13])
    hidden_layer_sizes = int(x[14])
    momentum = float(x[15])
    #activation = ['identity', 'relu'][int(x[16])]
    activation = 'identity'
    alpha_mlp = float(x[17])
    print(f"max_depth_rf: {max_depth_rf}, min_samples_split: {min_samples_split}, n_estimators_rf: {n_estimators_rf}, C: {C}, epsilon: {epsilon}, gamma: {gamma}, kernel: {kernel}, n_neighbors: {n_neighbors}, leaf_size: {leaf_size}, weights: {weights}, alpha: {alpha}, l1_ratio: {l1_ratio}, learning_rate: {learning_rate}, n_estimators_gb: {n_estimators_gb}, hidden_layers: {hidden_layer_sizes}, momentum: {momentum}, alpha_mlp: {alpha_mlp}, activation: {activation}")

    ensemble = Ensemble()
    ensemble.load_data()
    mre, r2, rmse = ensemble.StackingClassifier(x)
    return mre



def obj_fun2(x):

    ensemble = Ensemble()
    ensemble.load_data()
    mre, r2, rmse = ensemble.StackingClassifier(x)

    print("MRE:{}".format(mre))
    print("RMSE:{}".format(rmse))
    print("R2:{}".format(r2))




if __name__ == "__main__":
    # Bounds for parameters space
    lb = [2, 0.1, 10, 0.01, 0.01, 0.001, 1, 3, 10, 0, 0, 0, 0.01, 10, 10, 0.1, 0, 0.0001]
    ub = [32, 0.3, 400, 600, 1.1, 0.04, 2, 10, 55, 1, 1, 1, 0.05, 500, 100, 0.9, 1, 0.1]

    # Run the optimization
    xopt, fopt = pso(obj_fun, lb, ub, swarmsize=20, maxiter=40, debug=True)
    print('OPTIMAL PARAMETERS:')
    print(xopt, fopt)

    # Store the best params
    algo_params = {
        'max_depth_rf' : int(xopt[0]),
        'min_samples_split' : float(xopt[1]),
        'n_estimators_rf': int(xopt[2]),
        'C' : float(xopt[3]),
        'epsilon' : float(xopt[4]),
        'gamma' : float(xopt[5]),
        'kernel' : ['poly', 'rbf', 'sigmoid'][int(xopt[6])],
        'n_neighbors' : int(xopt[7]),
        'leaf_size' : int(xopt[8]),
        'weights' : ['uniform', 'distance'][int(xopt[9])],
        'alpha' : float(xopt[10]),
        'l1_ratio' : float(xopt[11]),
        'learning_rate' : float(xopt[12]),
        'n_estimators_gb' : int(xopt[13]),
        'hidden_layer_sizes' : int(xopt[14]),
        'momentum' : float(xopt[15]),
        'activation' : ['identity', 'relu'][int(xopt[16])],
        'alpha_mlp' : float(xopt[17]),
                   }

    print(algo_params)

    obj_fun2(xopt)

