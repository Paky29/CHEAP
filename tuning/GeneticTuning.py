import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, r2_score, make_scorer
from sklearn_genetic import GASearchCV
from sklearn_genetic.space import Continuous, Categorical, Integer
from sklearn_genetic.plots import plot_fitness_evolution, plot_search_space
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt

def my_custom_metric(y_true, y_pred):
    return r2_score(y_true, y_pred)

def gen_opt():

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

    # this library even performs well on weird ranges
    # adjust this to better ranges for better results
    param_grid = {
        'n_estimators': Integer(200, 2000),
        'max_features': Categorical(['sqrt', 'log2']),
        'max_depth': Integer(10, 1000),
        'min_samples_split': Integer(2, 14),
        'min_samples_leaf': Integer(1, 8),
        'criterion': Categorical(['absolute_error', 'friedman_mse', 'squared_error'])
    }

    # The base classifier to tune
    clf = RandomForestRegressor()

    # Our cross-validation strategy (it could be just an int)
    cv = StratifiedKFold(n_splits=3, shuffle=True)

    my_custom_scorer = make_scorer(my_custom_metric, greater_is_better=True)

    # The main class from sklearn-genetic-opt
    evolved_estimator = GASearchCV(estimator=clf,
                                  scoring=my_custom_scorer,
                                  param_grid=param_grid,
                                  n_jobs=-1,
                                  verbose=True,
                                  population_size=10,
                                  generations=10)

    # Train and optimize the estimator
    evolved_estimator.fit(X_train, y_train)

    # Best parameters found
    print(evolved_estimator.best_params_)
    # Use the model fitted with the best parameters
    y_predict_ga = evolved_estimator.predict(X_test)
    print(r2_score(y_test, y_predict_ga))


if __name__ == "__main__":
    gen_opt()
