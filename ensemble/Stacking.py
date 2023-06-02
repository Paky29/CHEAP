import numpy as np
from sklearn.linear_model import ElasticNet, LinearRegression, Lasso, Ridge
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
from deap import base, creator, tools
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from feature_selection import GeneticAlgorithm

#Stacking
#LassoRegression, RidgeRegression, ElasticNet

def evaluate(individual, X_train, X_test, y_train, y_test):
    selected_features = [feature for feature, mask in zip(X_train.columns, individual) if mask]
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]

    lasso_regressor = Lasso()
    lasso_regressor.fit(X_train_selected, y_train)

    ridge_regressor = Ridge()
    ridge_regressor.fit(X_train_selected, y_train)

    elasticnet_regressor = ElasticNet()
    elasticnet_regressor.fit(X_train_selected, y_train)

    #previsioni sul set di test
    y_pred_lasso = lasso_regressor.predict(X_test_selected)
    y_pred_ridge = ridge_regressor.predict(X_test_selected)
    y_pred_elasticnet = elasticnet_regressor.predict(X_test_selected)

    X_meta = np.column_stack((y_pred_lasso, y_pred_ridge, y_pred_elasticnet))

    meta_regressor = LinearRegression()
    meta_regressor.fit(X_meta, y_test)

    #previsioni sul set di test del meta-modello
    meta_pred = meta_regressor.predict(X_meta)

    #metrics
    rmse = np.sqrt(mean_squared_error(y_test, meta_pred))
    r2 = r2_score(y_test, meta_pred)
    mre = np.mean(np.abs(y_test - meta_pred) / y_test) * 100

    return rmse, r2, mre

def runStackingModels():
    print('\n--Stacking--\n-LinearRegression ,LassoRegression, RidgeRegression, ElasticNet e GradientBoostingRegression-\n')

    gb_regressor = GradientBoostingRegressor()
    gb_regressor.fit(X_train, y_train)

    lasso_regressor = Lasso()
    lasso_regressor.fit(X_train, y_train)

    ridge_regressor = Ridge()
    ridge_regressor.fit(X_train, y_train)

    elasticnet_regressor = ElasticNet()
    elasticnet_regressor.fit(X_train, y_train)

    #previsioni sul set di test
    y_pred_gb = gb_regressor.predict(X_test)
    y_pred_lasso = lasso_regressor.predict(X_test)
    y_pred_ridge = ridge_regressor.predict(X_test)
    y_pred_elasticnet = elasticnet_regressor.predict(X_test)

    X_meta = np.column_stack((y_pred_gb, y_pred_lasso, y_pred_ridge, y_pred_elasticnet))

    meta_regressor = LinearRegression()
    meta_regressor.fit(X_meta, y_test)

    #previsioni sul set di test del meta-modello
    meta_pred = meta_regressor.predict(X_meta)

    #model performance
    print('RMSE:', np.sqrt(mean_squared_error(y_test, meta_pred)))
    print('R^2 score:', r2_score(y_test, meta_pred))
    print('MRE:', np.mean(np.abs(y_test - meta_pred) / y_test) * 100)
    print('-------------------------')

def runFeatureSelection():
    #inizializzazione DEAP
    creator.create("FitnessMax", base.Fitness, weights=(-1.0, -1.0, -1.0))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", np.random.choice, [False, True])
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, len(X.columns))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluate, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    GeneticAlgorithm.runGA(toolbox, X)

#main
if __name__ == '__main__':
    seera = pd.read_csv("../datasets/SEERA.csv", delimiter=',', decimal=".")
    X = seera.drop('Effort', axis=1)
    y = seera['Effort']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
    #runFeatureSelection()
    runStackingModels()