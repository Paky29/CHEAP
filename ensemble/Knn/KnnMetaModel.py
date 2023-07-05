import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import smogn
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Lars

class Ensemble:
    def __init__(self):
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.k = 6

        self.X = None
        self.y = None

        self.mre = None
        self.rmse = None
        self.r2 = None
        self.train_acc = None
        self.test_acc = None

    def data_balancing(self, X, y):

        train = X.copy()
        train.loc[:, 'Actual effort'] = y
        train = train.reset_index(drop=True)
        train_smogn = smogn.smoter(
            data=train,
            y='Actual effort',
            k=len(X),  ## positive integer (k < n)
            samp_method='extreme',  ## ('balance' or 'extreme')
            # under_samp=True,

            ## phi relevance arguments
            rel_thres=0.85,  ## positive real number (0 < R < 1)
            rel_method='auto',  ## string ('auto' or 'manual')
            rel_xtrm_type='both',  ## string ('low' or 'both' or 'high')
            rel_coef=2.30
        )

        return train_smogn.drop(['Actual effort'], axis=1), train_smogn['Actual effort']

    def printMetrics(self):
        print("-------------------------------")
        print("\nMetaModel metrics:")
        print(f"RMSE: {self.rmse}")
        print(f"MRE: {self.mre}")
        print(f"R2: {self.r2}")
        print(f"Train Accuracy: {self.train_acc}")
        print(f"Test Accuracy: {self.test_acc}")
        print("-------------------------------")

    def load_data(self):
        seera = pd.read_csv("../../datasets/SEERA_train.csv", delimiter=',', decimal=".")
        seera.drop("Indice Progetto",axis=1)
        self.X = seera.drop(['Actual effort'], axis=1)
        self.y = seera['Actual effort']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.15,
                                                                                random_state=23)

    def StackingRegressor(self):

        rf_regressor = RandomForestRegressor()
        #gb_regressor = GradientBoostingRegressor()
        #linear = LinearRegression()
        ada = AdaBoostRegressor(n_estimators = 78, learning_rate = 0.011444827305475122, loss= 'exponential')
        #xgb = XGBRegressor(colsample_bytree=0.41470186669055753, gamma=0.06493204581815717, learning_rate=0.028343176856406638, max_depth=3, n_estimators=630, min_child_weight=1, subsample = 0.876091138265439, reg_lambda = 0.40025015210995907)
        elastic = ElasticNet(alpha=0.9844970636892896, l1_ratio=0.10523697586301965,selection='random', tol=0.0009844912834425824)
        svr = SVR(C=4.986086586865859, epsilon=0.025578671993318654,gamma=0.029267910098574566, kernel='linear')
        #lars = make_pipeline(RobustScaler(),Lars(n_nonzero_coefs=3))
        # Define weak learners
        weak_learners = [
            ('rf', rf_regressor),
            ('ada', ada),
            ('svr',svr),
            ('en', elastic)
        ]

        final_learner = make_pipeline(MinMaxScaler(feature_range=(-1,1)),KNeighborsRegressor(n_neighbors=7))


        train_meta_model = None
        test_meta_model = None

        # Start stacking
        for clf_id, clf in weak_learners:
            # Predictions for each regressor based on k-fold
            predictions_clf = self.k_fold_cross_validation(clf)
            print("*******************************")

            # Predictions for test set for each regressor based on train of level 0
            test_predictions_clf = self.train_level_0(clf)

            # Stack predictions which will form the input data for the data model
            if isinstance(train_meta_model, np.ndarray):
                train_meta_model = np.vstack((train_meta_model, predictions_clf))
            else:
                train_meta_model = predictions_clf

            # Stack predictions from test set which will form test data for metamodel
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
        self.printMetrics()

    def k_fold_cross_validation(self, clf):
        k_predictions = None

        # Number of samples per fold
        folders_size = int(len(self.X_train) / self.k)

        # Stars k-fold cross validation
        for fold in range(self.k):

            # Settings for each folders_size
            if fold == (self.k - 1):
                batch_start = folders_size * fold
                batch_finish = self.X_train.shape[0]
            else:
                batch_start = folders_size * fold
                batch_finish = folders_size * (fold + 1)

            # test & training samples for each fold iteration
            fold_X_test = self.X_train.iloc[batch_start:batch_finish, :]
            fold_x_train = self.X_train.iloc[[index for index in range(self.X_train.shape[0]) if
                                              index not in range(batch_start, batch_finish)], :]

            # test & training targets for each fold iteration
            fold_y_test = self.y_train.iloc[batch_start:batch_finish]
            fold_y_train = self.y_train.iloc[
                [index for index in range(self.X_train.shape[0]) if index not in range(batch_start, batch_finish)]]

            x_bal, y_bal = self.data_balancing(fold_x_train, fold_y_train)

            # Fit current regressor
            clf.fit(x_bal, y_bal)
            fold_y_pred = clf.predict(fold_X_test)

            print(fold + 1, " iterazione kfold")
            print("r2: ", r2_score(fold_y_test, fold_y_pred))
            print("rmse: ", np.sqrt(mean_squared_error(fold_y_test, fold_y_pred)))
            print("mre: ", np.mean(np.abs(fold_y_test - fold_y_pred) / fold_y_test) * 100)

            # Store predictions for each fold_X_test
            if isinstance(k_predictions, np.ndarray):
                k_predictions = np.concatenate((k_predictions, fold_y_pred))
            else:
                k_predictions = fold_y_pred

        return k_predictions

    def train_level_0(self, clf):

        clf.fit(self.X_train, self.y_train)
        y_pred = clf.predict(self.X_test)

        trainR2 = clf.score(self.X_train, self.y_train)
        testR2 = clf.score(self.X_test, self.y_test)

        #print("\nModel: ", clf.steps[-1][1].__class__.__name__)

        print("\nModel: ", clf)
        print("rmse: ", np.sqrt(mean_squared_error(self.y_test, y_pred)))
        print("r2 (su test): ", r2_score(self.y_test, y_pred))
        print("mre: ", np.mean(np.abs(self.y_test - y_pred) / self.y_test) * 100)
        print("Train r2: ", trainR2)

        return y_pred

    def train_level_1(self, final_learner, train_meta_model, test_meta_model):

        final_learner.fit(train_meta_model, self.y_train)

        predictions = final_learner.predict(test_meta_model)

        print(predictions)
        print(self.y_test)

        self.rmse = np.sqrt(mean_squared_error(self.y_test, predictions))
        self.r2 = r2_score(self.y_test, predictions)
        self.mre = np.mean(np.abs(self.y_test - predictions) / self.y_test) * 100
        self.train_acc = final_learner.score(train_meta_model, self.y_train)
        self.test_acc = final_learner.score(test_meta_model, self.y_test)


if __name__ == "__main__":
    ensemble = Ensemble()
    ensemble.load_data()
    ensemble.StackingRegressor()
