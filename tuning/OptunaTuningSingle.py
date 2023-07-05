import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import smogn
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR


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

    def load_data(self):
        seera = pd.read_csv("SEERA_retrain.csv", delimiter=',', decimal=".")
        self.y = seera['Actual effort']

        self.X = seera[
            ['Customer organization type', 'Estimated  duration', 'Application domain', 'Government policy impact',
             'Organization management structure clarity', 'Developer training', 'Development team management',
             'Top management support', 'Top management opinion of previous system', 'User resistance',
             ' Requirements flexibility ', 'Consultant availability', 'DBMS  expert availability',
             'Software tool experience', 'Team size', 'Team contracts', 'Development environment adequacy',
             'Tool availability ', 'DBMS used', 'Degree of software reuse ', 'Degree of risk management',
             'Requirement accuracy level', 'Technical documentation', 'Required reusability',
             'Performance requirements',
             'Reliability requirements']]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.15,
                                                                                random_state=23)
        self.X_train, self.x_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train,
                                                                              test_size=0.20, random_state=23)
        self.X_bal, self.y_bal = self.data_balancing(self.X_train, self.y_train)

    def BlendingRegressor(self, params, clf_tune):

        # Initialize the weak learners
        rf_regressor = RandomForestRegressor()
        ada = AdaBoostRegressor()
        elastic = ElasticNet()
        svr = SVR()

        # Define weak learners
        weak_learners = [
            ('rf', rf_regressor),
            ('ada', ada),
            ('svr', svr),
            ('en', elastic)
        ]

        # Define the final learner
        knn = KNeighborsRegressor(n_neighbors=11, leaf_size=5, weights='distance', p=1)

        if clf_tune == 'knn':
            knn.set_params(**params)

        final_learner = make_pipeline(MinMaxScaler(), knn)

        train_meta_model = None
        test_meta_model = None

        # Start stacking
        for clf_id, clf in weak_learners:

            # Set suggested params for learner to tune
            if clf_id == clf_tune:
                clf.set_params(**params)

            # Predictions for each classifier based on k-fold
            val_predictions, test_predictions, mre = self.train_level_0(clf)

            if clf_id == clf_tune:
                clf_mre = mre

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

        # Return metric to optimize
        return self.mre if clf_tune == 'knn' else clf_mre

    def train_level_0(self, clf):
        clf.fit(self.X_bal, self.y_bal)

        val_predictions = clf.predict(self.x_val)
        test_predictions = clf.predict(self.X_test)

        print("\nModel: ", clf)

        r2 = r2_score(self.y_val, val_predictions)
        rmse = np.sqrt(mean_squared_error(self.y_val, val_predictions))
        mre = np.mean(np.abs(self.y_val - val_predictions) / self.y_val) * 100
        print("Fval-R2: " + str(r2))
        print("Fval-MRE: " + str(mre))
        print("Fval-RMSE: " + str(rmse))
        print("")

        # Generate predictions for original test set
        # These predictions will be used to test the meta model
        r2_t = r2_score(self.y_test, test_predictions)
        rmse_t = np.sqrt(mean_squared_error(self.y_test, test_predictions))
        mre_t = np.mean(np.abs(self.y_test - test_predictions) / self.y_test) * 100
        print("Test-R2: " + str(r2_t))
        print("Test-MRE: " + str(mre_t))
        print("Test-RMSE: " + str(rmse_t))

        return val_predictions, test_predictions, mre_t

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


def objective(trial, clf_tune):
    if clf_tune == 'rf':
        params = {
            'n_estimators': trial.suggest_int("rf_n_estimators", 5, 100),
            'max_features': trial.suggest_categorical('rf_max_features', ['auto', 'sqrt']),
            'max_depth': trial.suggest_int("rf_max_depth", 10, 120),
            'min_samples_split': trial.suggest_int('rf_min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('rf_min_samples_leaf', 1, 4)
        }

    elif clf_tune == 'ada':
        params = {
            'n_estimators': trial.suggest_int("n_estimators", 70, 90),
            'learning_rate': trial.suggest_float("learning_rate", 0.01, 0.1),
            'loss': trial.suggest_categorical('loss', ['linear', 'square', 'exponential'])
        }

    elif clf_tune == 'svr':
        params = {
            'C': trial.suggest_float("svr_C", 0.1, 5.0),
            'epsilon': trial.suggest_float("svr_epsilon", 0.01, 0.5),
            'gamma': trial.suggest_float("svr_gamma", 0.001, 0.1)
        }

    elif clf_tune == 'en':
        params = {
            'alpha': trial.suggest_float("en_alpha", 0.01, 1.0),
            'l1_ratio': trial.suggest_float("en_l1_ratio", 0.1, 1.0),
            'tol': trial.suggest_float("en_tol", 0.0001, 0.001)
        }

    else:
        params = {
            'n_neighbors': trial.suggest_int("knn_n_neighbors", 5, 15),
            'leaf_size': trial.suggest_int("knn_leaf_size", 1, 30),
            'weights': trial.suggest_categorical("knn_weights", ['uniform', 'distance']),
            'p': trial.suggest_int("knn_p", 1, 2)
        }

    ensemble = Ensemble()

    try:
        ensemble.load_data()
        mre = ensemble.BlendingRegressor(params, clf_tune)
    except Exception as inst:
        mre = 10000

    return mre


def optuna_single_tuning():

    clf_tune = 'knn'

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


if __name__ == '__main__':
    optuna_single_tuning()
