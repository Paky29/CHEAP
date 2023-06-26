import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import ElasticNet


class Ensemble:
    def __init__(self):
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.k = 10

    def load_data(self):
        seera = pd.read_csv("datasets/SEERA_train.csv", delimiter=',', decimal=".")
        X = seera.drop(['Indice Progetto','Actual effort'],axis=1)
        y = seera['Actual effort']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=23)
        print(self.X_train.iloc[0:12,:])

    def StackingClassifier(self):

        rf_regressor = RandomForestRegressor(max_depth=16, min_samples_split=0.1810111262255957, n_estimators=108)
        svr = SVR(C=85.31509250743692, epsilon=0.8345275221470401, gamma=0.010112870522012395, kernel='rbf')

        elasticnet_regressor = ElasticNet(alpha=0.729511921784814, l1_ratio=0.46870792837162806)

        knn_regressor = KNeighborsRegressor(n_neighbors=5, leaf_size=21, weights='uniform')

        gb_regressor = GradientBoostingRegressor(learning_rate=0.021383363768917734, n_estimators=154)

        # Define weak learners
        weak_learners = [('rf', rf_regressor),
                         ('knn', knn_regressor),
                         ('svr', svr),
                         ('gb', gb_regressor),
                         ('en', elasticnet_regressor)]

        # Final learner or meta model
        final_learner = MLPRegressor(random_state=1, max_iter=500)

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
        self.train_level_1(final_learner, train_meta_model, test_meta_model)

    def k_fold_cross_validation(self, clf):

        predictions_clf = None

        # Number of samples per fold
        batch_size = int(len(self.X_train) / self.k)
        print(batch_size)

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
        print(f"Test r2_nostro: {r2}")
        print(f"Train r2: {final_learner.score(train_meta_model, self.y_train)}")
        print(f"Test r2: {final_learner.score(test_meta_model, self.y_test)}")


if __name__ == "__main__":
    ensemble = Ensemble()
    ensemble.load_data()
    ensemble.StackingClassifier()