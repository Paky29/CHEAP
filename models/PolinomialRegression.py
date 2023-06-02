import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
from deap import base, creator, tools
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from feature_selection import GeneticAlgorithm


def evaluate(individual, X_train, X_test, y_train, y_test):
    selected_features = [feature for feature, mask in zip(X_train.columns, individual) if mask]
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]

    # Creazione delle feature polinomiali
    poly_transformer = PolynomialFeatures(degree=2)  # Imposta il grado desiderato
    X_train_poly = poly_transformer.fit_transform(X_train_selected)
    X_test_poly = poly_transformer.transform(X_test_selected)

    # Addestramento del modello di regressione lineare
    poly_regressor = LinearRegression()
    poly_regressor.fit(X_train_poly, y_train)

    # Effettua le previsioni
    y_pred = poly_regressor.predict(X_test_poly)

    #metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mre = np.mean(np.abs(y_test - y_pred) / y_test) * 100

    return rmse, r2, mre

def runPolinomialRegression():
    print('\n--PolinomialRegression--')

    # Genera le feature polinomiali
    poly_features = PolynomialFeatures(degree=2)  # Imposta il grado del polinomio desiderato
    X_train_poly = poly_features.fit_transform(X_train)
    X_test_poly = poly_features.transform(X_test)

    # Esegui una regressione lineare sulle feature polinomiali
    linear_regressor = LinearRegression()
    linear_regressor.fit(X_train_poly, y_train)

    # Fai le previsioni sui dati di test
    y_pred = linear_regressor.predict(X_test_poly)

    # Evaluate the model performance
    print('RMSE:', np.sqrt(mean_squared_error(y_test, y_pred)))
    print('R^2 score:', r2_score(y_test, y_pred))
    print('MRE:', np.mean(np.abs(y_test - y_pred) / y_test) * 100)
    print('-------------------------')

def runFeatureSelection():
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

#main
if __name__ == '__main__':
    seera = pd.read_csv("../datasets/SEERA.csv", delimiter=',', decimal=".")
    X = seera.drop('Effort', axis=1)
    y = seera['Effort']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

    runFeatureSelection()

