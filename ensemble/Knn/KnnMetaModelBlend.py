import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import smogn
from sklearn.linear_model import ElasticNet
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
from sklearn.feature_selection import SelectKBest, f_regression, SequentialFeatureSelector, RFE, SelectFromModel
from sklearn.neighbors import KNeighborsRegressor

class Ensemble:
    def __init__(self):
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.X = None
        self.y = None

        self.mre = None
        self.rmse = None
        self.r2 = None

        self.y_bal = None
        self.X_bal = None

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


    def load_data(self):
        seera = pd.read_csv("../../datasets/SEERA_retrain.csv", delimiter=',', decimal=".")
        self.y = seera['Actual effort']
        #self.X = seera.drop('Actual effort',axis = 1)
        self.X = seera[['Customer organization type', 'Estimated  duration', 'Application domain', 'Government policy impact',
          'Organization management structure clarity', 'Developer training', 'Development team management',
          'Top management support', 'Top management opinion of previous system',
          ' Requirements flexibility ', 'Consultant availability', 'DBMS  expert availability',
          'Software tool experience', 'Team size', 'Team contracts', 'Development environment adequacy',
          'Tool availability ', 'DBMS used', 'Degree of software reuse ', 'Degree of risk management',
          'Requirement accuracy level', 'Technical documentation', 'Required reusability', 'Performance requirements',
          'Reliability requirements']]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.15, random_state=23)
        self.X_train, self.x_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train, test_size=0.20, random_state=23)
        self.X_bal, self.y_bal = self.data_balancing(self.X_train, self.y_train)

    def BlendingRegressor(self):
        rf_regressor = RandomForestRegressor(max_depth=67, max_features=1.0, n_estimators=67)
        ada = AdaBoostRegressor(learning_rate=0.05984073241086173, n_estimators=77)
        elastic = ElasticNet(alpha=0.9844970636892896, l1_ratio=0.10523697586301965, selection='random', tol=0.0009844912834425824)
        svr = SVR(C=4.973571744211175, epsilon=0.24236503269448004, gamma=0.06602126901315636, kernel='linear')

        # Define weak learners
        weak_learners = [
            ('rf', rf_regressor),
            ('ada', ada),
            ('svr', svr),
            ('en', elastic)
        ]

        final_learner = make_pipeline(MinMaxScaler(),KNeighborsRegressor(n_neighbors = 11, leaf_size = 22, weights = 'distance', p = 2))

        train_meta_model = None
        test_meta_model = None

        # Start stacking
        for clf_id, clf in weak_learners:
            # Predictions for each classifier based on k-fold
            val_predictions, test_predictions = self.train_level_0(clf)

            # Stack predictions which will form the input data for the data model
            if isinstance(train_meta_model, np.ndarray):
                train_meta_model = np.vstack((train_meta_model, val_predictions))
            else:
                train_meta_model = val_predictions

            # Stack predictions from test set
            # which will form test data for meta model
            if isinstance(test_meta_model, np.ndarray):
                test_meta_model = np.vstack((test_meta_model, test_predictions))
            else:
                test_meta_model = test_predictions

        # Transpose train_meta_model
        train_meta_model = train_meta_model.T

        # Transpose test_meta_model
        test_meta_model = test_meta_model.T

        # Training level 1
        self.train_level_1(final_learner, train_meta_model, test_meta_model)

    def train_level_0(self, clf):
        clf.fit(self.X_bal, self.y_bal)

        val_predictions = clf.predict(self.x_val)
        test_predictions = clf.predict(self.X_test)

        r2 = r2_score(self.y_val, val_predictions)
        print("Fvalidazione "+str(r2))

        # Generate predictions for original test set
        # These predictions will be used to test the meta model
        r2 = r2_score(self.y_test, test_predictions)
        print("Test "+str(r2))

        return val_predictions, test_predictions

    def train_level_1(self, final_learner, train_meta_model, test_meta_model):

        final_learner.fit(train_meta_model, self.y_val)
        predictions = final_learner.predict(test_meta_model)

        self.rmse = np.sqrt(mean_squared_error(self.y_test, predictions))
        self.r2 = r2_score(self.y_test, predictions)
        self.mre = np.mean(np.abs(self.y_test - predictions) / self.y_test) * 100

        print(predictions)
        print(self.y_test)

        print(f"Test rmse: {self.rmse}")
        print(f"Test mre: {self.mre}")
        print(f"Test r2: {self.r2}")

        # Getting train and test accuracies from meta_model
        print(f"Train accuracy: {final_learner.score(train_meta_model, self.y_val)}")

if __name__ == "__main__":
    ensemble = Ensemble()
    ensemble.load_data()
    ensemble.BlendingRegressor()
