import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from deap import base, creator, tools
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from feature_selection import GeneticAlgorithm

def evaluate(individual, X_train, X_test, y_train, y_test):
    selected_features = [feature for feature, mask in zip(X_train.columns, individual) if mask]
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]

    knn_regressor = KNeighborsRegressor(n_neighbors=5)
    knn_regressor.fit(X_train_selected, y_train)
    y_pred = knn_regressor.predict(X_test_selected)

    #metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mre = np.mean(np.abs(y_test - y_pred) / y_test) * 100

    return rmse, r2, mre

def run(X,y):
    print('KNearestNeighbors')
    # split 70/30
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

    knn_regressor = KNeighborsRegressor(n_neighbors=5)
    knn_regressor.fit(X_train, y_train)

    y_pred = knn_regressor.predict(X_test)
    # Evaluate the model performance
    print('RMSE:', np.sqrt(mean_squared_error(y_test, y_pred)))
    print('R^2 score:', r2_score(y_test, y_pred))
    print('MRE:', np.mean(np.abs(y_test - y_pred) / y_test) * 100)
    print('-------------------------')

def runFeatureSelection(X,y):
    #split 70/30
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

    # Inizializzazione DEAP
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