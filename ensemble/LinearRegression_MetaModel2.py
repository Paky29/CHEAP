import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from deap import base, creator, tools
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from feature_selection import GeneticAlgorithm

#Models: Random Forest, Gradient Boosting, knn
#Meta: Linear Regression

def evaluate(individual, X_train, X_test, y_train, y_test):
    selected_features = [feature for feature, mask in zip(X_train.columns, individual) if mask]
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]

    #random forest
    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_regressor.fit(X_train_selected, y_train)

    #gradientboosting
    gb_regressor = GradientBoostingRegressor()
    gb_regressor.fit(X_train_selected, y_train)

    #KNN
    knn_regressor = KNeighborsRegressor(n_neighbors=5)
    knn_regressor.fit(X_train_selected, y_train)


    #previsioni sul set di test
    y_pred_rf = rf_regressor.predict(X_test_selected)
    y_pred_gb = gb_regressor.predict(X_test_selected)
    y_pred_knn = knn_regressor.predict(X_test_selected)

    X_meta = np.column_stack((y_pred_rf, y_pred_gb, y_pred_knn))

    meta_regressor = LinearRegression()
    meta_regressor.fit(X_meta, y_test)

    #previsioni sul set di test del meta-modello
    meta_pred = meta_regressor.predict(X_meta)

    #metrics
    rmse = np.sqrt(mean_squared_error(y_test, meta_pred))
    r2 = r2_score(y_test, meta_pred)
    mre = np.mean(np.abs(y_test - meta_pred) / y_test) * 100

    return rmse, r2, mre

def run(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
    print('Models: Random Forest, Knn, Gradient Boosting\nMetaModel: Linear Regression')

    # random forest
    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_regressor.fit(X_train, y_train)

    # gradientboosting
    gb_regressor = GradientBoostingRegressor()
    gb_regressor.fit(X_train, y_train)

    # KNN
    knn_regressor = KNeighborsRegressor(n_neighbors=5)
    knn_regressor.fit(X_train, y_train)

    #previsioni sul set di test
    y_pred_rf = rf_regressor.predict(X_test)
    y_pred_gb = gb_regressor.predict(X_test)
    y_pred_knn = knn_regressor.predict(X_test)

    X_meta = np.column_stack((y_pred_gb, y_pred_rf, y_pred_knn))

    meta_regressor = LinearRegression()
    meta_regressor.fit(X_meta, y_test)

    #previsioni sul set di test del meta-modello
    meta_pred = meta_regressor.predict(X_meta)

    #model performance
    print('RMSE:', np.sqrt(mean_squared_error(y_test, meta_pred)))
    print('R^2 score:', r2_score(y_test, meta_pred))
    print('MRE:', np.mean(np.abs(y_test - meta_pred) / y_test) * 100)
    print('-------------------------')

def runFeatureSelection(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
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