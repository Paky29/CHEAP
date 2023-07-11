import matplotlib.pyplot as plt
import numpy as np
import optuna
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
from sklearn.svm import SVR
from xgboost import XGBRegressor


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

    def arrotonda(self,dataset):
      # Arrotonda tutti i valori del dataset
      decimali = 2  # Numero di decimali desiderati

      dataset = dataset.round(decimali)

      dataset = np.rint(dataset)

      return dataset

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
          rel_thres=0.70,  ## positive real number (0 < R < 1)
          rel_method='auto',  ## string ('auto' or 'manual')
          rel_xtrm_type='both',  ## string ('low' or 'both' or 'high')
          rel_coef=2.00
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
        seera = pd.read_csv("SEERA_nofs.csv", delimiter=',', decimal=".")
        self.X = seera[['Organization type', 'Estimated  duration', 'Development type', 'Government policy impact', 'Developer hiring policy', 'Developer incentives policy ', 'Development team management', 'Consultant availability', 'DBMS  expert availability', 'Software tool experience', 'Programmers experience in programming language', 'Team size', 'Daily working hours', 'Team contracts', 'Schedule quality', 'Development environment adequacy', 'Tool availability ', 'Methodology', 'Degree of software reuse ', 'Requirement accuracy level', 'User manual', 'Performance requirements']]
        self.y = seera['Actual effort']
        #self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.20,random_state=42)

    def StackingRegressor(self, params):

        # Initialize the weak learners
        params_rf = params[0]
        params_ada = params[1]
        params_svr = params[2]
        params_en = params[3]
        params_knn = params[4]

        rf_regressor = RandomForestRegressor(**params_rf)
        ada = AdaBoostRegressor(**params_ada)
        elastic = ElasticNet(**params_en)
        svr = SVR(**params_svr)

        # Define weak learners
        weak_learners = [
            ('rf', rf_regressor),
            ('ada', ada),
            ('svr', svr),
            ('en', elastic)
        ]


        # Define the final learner
        knn = KNeighborsRegressor(**params_knn)

        final_learner = make_pipeline(MinMaxScaler(feature_range=(-1,1)), knn)

        train_meta_model = None
        test_meta_model = None
        kfold = KFold(n_splits=8, shuffle=True, random_state=23)

        # Liste per memorizzare le performance dei modelli in ogni fold
        rmse_scores = []
        r2_scores = []
        mre_scores = []


        # Suddivisione dei dati utilizzando la k-fold cross-validation
        for train_indices, test_indices in kfold.split(self.X):
            self.X_train = self.X.iloc[train_indices]
            self.X_test = self.X.iloc[test_indices]
            self.y_train = self.y.iloc[train_indices]
            self.y_test = self.y.iloc[test_indices]
            self.X_train, self.y_train = self.data_balancing(self.X_train, self.y_train)

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

            rmse_scores.append(self.rmse)
            r2_scores.append(self.r2)
            mre_scores.append(self.mre)

            train_meta_model = None
            test_meta_model = None

        mean_rmse = np.mean(rmse_scores)
        mean_r2 = np.mean(r2_scores)
        mean_mre = np.mean(mre_scores)


        # Print average performances

        print('Average RMSE:', mean_rmse)
        print('Average R^2 score:', mean_r2)
        print('Average MRE:', mean_mre)
        print('-------------------------')

        return mean_mre

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


            # Fit current regressor
            clf.fit(fold_x_train, fold_y_train)
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


def objective(trial):

    params_rf = {
          'n_estimators' : trial.suggest_int("rf_n_estimators", 5, 100),
          'max_features' : trial.suggest_categorical('rf_max_features', ['auto', 'sqrt']),
          'max_depth' : trial.suggest_int("rf_max_depth", 10, 120),
          'min_samples_split' : trial.suggest_int('rf_min_samples_split', 2, 10),
          'min_samples_leaf' : trial.suggest_int('rf_min_samples_leaf', 1, 4)
      }

    params_ada = {
        'n_estimators' : trial.suggest_int("ada_n_estimators", 70, 90),
        'learning_rate' : trial.suggest_float("learning_rate", 0.01, 0.1),
        'loss' : trial.suggest_categorical('loss', ['linear', 'square', 'exponential'])
    }

    params_svr = {
        'C' : trial.suggest_float("svr_C", 0.1, 5.0),
        'epsilon' : trial.suggest_float("svr_epsilon", 0.01, 0.5),
        'gamma' : trial.suggest_float("svr_gamma", 0.001, 0.1)
    }

    params_en= {
        'alpha' : trial.suggest_float("en_alpha", 0.01, 1.0),
        'l1_ratio' : trial.suggest_float("en_l1_ratio", 0.1, 1.0),
        'tol' : trial.suggest_float("en_tol", 0.0001, 0.001)
    }

    params_knn = {
        'n_neighbors' : trial.suggest_int("knn_n_neighbors", 5, 15),
        'leaf_size' : trial.suggest_int("knn_leaf_size", 1, 30),
        'weights' : trial.suggest_categorical("knn_weights", ['uniform', 'distance']),
        'p' : trial.suggest_int("knn_p", 1, 2)
    }

    params = []
    params.append(params_rf)
    params.append(params_ada)
    params.append(params_svr)
    params.append(params_en)
    params.append(params_knn)


    ensemble = Ensemble()

    try:
      ensemble.load_data()
      mre = ensemble.StackingRegressor(params)
    except Exception as inst:
      mre = 10000

    return mre


def optuna_multiple_tuning():

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)

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
    optuna_multiple_tuning()
