import numpy as np
from joblib import dump
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from deap import base, creator, tools
from sklearn.model_selection import KFold, train_test_split
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

from data_cleaning.cleanSeera import save_dataset, scaling_robust
from feature_selection import GeneticAlgorithm

#Stacking
#Random Forest, SVR, Gradient Boosting, knn, Elastic Net

def save_model(model, filename):
    # Save the model to a file
    dump(model, "models_saved/"+filename)

def evaluate(individual, X, y):
    selected_features = [feature for feature, mask in zip(X.columns, individual) if mask]
    X_selected = X[selected_features]

    # Creazione dell'oggetto KFold con k=10
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)

    # Liste per memorizzare le performance dei modelli in ogni fold
    rmse_scores = []
    r2_scores = []
    mre_scores = []

    # Suddivisione dei dati utilizzando la k-fold cross-validation
    for train_indices, test_indices in kfold.split(X_selected):
        X_train = X_selected.iloc[train_indices]
        X_test = X_selected.iloc[test_indices]
        y_train = y.iloc[train_indices]
        y_test = y.iloc[test_indices]

        # Random Forest
        rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_regressor.fit(X_train, y_train)

        # ElasticNet
        elasticnet_regressor = ElasticNet()
        elasticnet_regressor.fit(X_train, y_train)

        # SVR
        svr = SVR(kernel='sigmoid')
        svr.fit(X_train, y_train)

        # Gradient Boosting
        gb_regressor = GradientBoostingRegressor()
        gb_regressor.fit(X_train, y_train)

        # KNN
        knn_regressor = KNeighborsRegressor(n_neighbors=5)
        knn_regressor.fit(X_train, y_train)

        # Previsioni sul set di test
        y_pred_rf = rf_regressor.predict(X_test)
        y_pred_svr = svr.predict(X_test)
        y_pred_gb = gb_regressor.predict(X_test)
        y_pred_knn = knn_regressor.predict(X_test)
        y_pred_elastic = elasticnet_regressor.predict(X_test)

        X_meta = np.column_stack((y_pred_rf, y_pred_svr, y_pred_gb, y_pred_knn, y_pred_elastic))

        meta_regressor = LinearRegression(fit_intercept=True, copy_X=True, positive=True)
        meta_regressor.fit(X_meta, y_test)

        # Previsioni sul set di test del meta-modello
        meta_pred = meta_regressor.predict(X_meta)

        # Calcolo delle misure di performance
        rmse = np.sqrt(mean_squared_error(y_test, meta_pred))
        r2 = r2_score(y_test, meta_pred)
        mre = np.mean(np.abs(y_test - meta_pred) / y_test) * 100

        # Aggiunta delle performance alle liste
        rmse_scores.append(rmse)
        r2_scores.append(r2)
        mre_scores.append(mre)

    # Calcolo delle medie delle performance
    mean_rmse = np.mean(rmse_scores)
    mean_r2 = np.mean(r2_scores)
    mean_mre = np.mean(mre_scores)

    return mean_rmse, mean_r2, mean_mre

def run_kfold(X,y):
    print('Models: Random Forest, SVR, ElasticNet, Knn, GradientBoosting\nMetaModel: LinearRegression')

    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    svr = SVR(kernel='sigmoid')

    elasticnet_regressor = ElasticNet()

    knn_regressor = KNeighborsRegressor(n_neighbors=5)

    gb_regressor = GradientBoostingRegressor()

    # Creazione dell'oggetto KFold con k=10
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)

    # Liste per memorizzare le performance dei modelli
    rmse_scores = []
    r2_scores = []
    mre_scores = []

    # Suddivisione dei dati utilizzando la k-fold cross-validation
    for train_indices, test_indices in kfold.split(X):
        X_train = X.iloc[train_indices]
        X_test = X.iloc[test_indices]
        y_train = y.iloc[train_indices]
        y_test = y.iloc[test_indices]

        rf_regressor.fit(X_train, y_train)
        #save_model(rf_regressor, "random_forest.joblib")
        svr.fit(X_train, y_train)
        #save_model(svr, "svr.joblib")
        gb_regressor.fit(X_train, y_train)
        #save_model(gb_regressor, "gradient_boosting.joblib")
        knn_regressor.fit(X_train, y_train)
        #save_model(knn_regressor, "knn.joblib")
        elasticnet_regressor.fit(X_train, y_train)
        #save_model(elasticnet_regressor, "elastic_net.joblib")

        # Previsioni sul set di test
        y_pred_rf = rf_regressor.predict(X_test)
        y_pred_svr = svr.predict(X_test)
        y_pred_gb = gb_regressor.predict(X_test)
        y_pred_knn = knn_regressor.predict(X_test)
        y_pred_elastic = elasticnet_regressor.predict(X_test)

        X_meta = np.column_stack((y_pred_gb, y_pred_rf, y_pred_svr, y_pred_knn, y_pred_elastic))

        meta_regressor = LinearRegression()
        meta_regressor.fit(X_meta, y_test)

        # Previsioni sul set di test del meta-modello
        meta_pred = meta_regressor.predict(X_meta)

        # Calcolo delle misure di performance
        rmse = np.sqrt(mean_squared_error(y_test, meta_pred))
        r2 = r2_score(y_test, meta_pred)
        mre = np.mean(np.abs(y_test - meta_pred) / y_test) * 100

        # Aggiunta delle performance alle liste
        rmse_scores.append(rmse)
        r2_scores.append(r2)
        mre_scores.append(mre)

    # Calcolo delle medie delle performance
    mean_rmse = np.mean(rmse_scores)
    mean_r2 = np.mean(r2_scores)
    mean_mre = np.mean(mre_scores)

    # Stampa delle performance medie
    print('Average RMSE:', mean_rmse)
    print('Average R^2 score:', mean_r2)
    print('Average MRE:', mean_mre)
    print('-------------------------')

def run(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.08, random_state=42)
    #X_train = X_train.drop('Indice Progetto',axis =1)
    #X_test = X_test.drop('Indice Progetto', axis=1)
    print('Models: Random Forest, Knn, Gradient Boosting\nMetaModel: Linear Regression')

    rf_regressor = RandomForestRegressor(max_depth= 16, min_samples_split= 0.1810111262255957, n_estimators = 108)
    svr = SVR(C = 85.31509250743692, epsilon = 0.8345275221470401, gamma = 0.010112870522012395, kernel = 'rbf')

    elasticnet_regressor = ElasticNet(alpha = 0.729511921784814, l1_ratio = 0.46870792837162806)

    knn_regressor = KNeighborsRegressor(n_neighbors=5, leaf_size = 21, weights = 'uniform')

    gb_regressor = GradientBoostingRegressor(learning_rate = 0.021383363768917734, n_estimators = 154)

    rf_regressor.fit(X_train, y_train)
    save_model(rf_regressor, "random_forest.joblib")
    svr.fit(X_train, y_train)
    save_model(svr, "svr.joblib")
    gb_regressor.fit(X_train, y_train)
    save_model(gb_regressor, "gradient_boosting.joblib")
    knn_regressor.fit(X_train, y_train)
    save_model(knn_regressor, "knn.joblib")
    elasticnet_regressor.fit(X_train, y_train)
    save_model(elasticnet_regressor, "elastic_net.joblib")

    # Previsioni sul set di test
    y_pred_rf = rf_regressor.predict(X_test)
    y_pred_svr = svr.predict(X_test)
    y_pred_gb = gb_regressor.predict(X_test)
    y_pred_knn = knn_regressor.predict(X_test)
    y_pred_elastic = elasticnet_regressor.predict(X_test)

    X_meta = np.column_stack((y_pred_gb, y_pred_rf, y_pred_svr, y_pred_knn, y_pred_elastic))

    meta_regressor = LinearRegression()
    meta_regressor.fit(X_meta, y_test)
    save_model(meta_regressor, "meta_regressor.joblib")

    # Previsioni sul set di test del meta-modello
    meta_pred = meta_regressor.predict(X_meta)

    # model performance
    print('RMSE:', np.sqrt(mean_squared_error(y_test, meta_pred)))
    print('R^2 score:', r2_score(y_test, meta_pred))
    print('MRE:', np.mean(np.abs(y_test - meta_pred) / y_test) * 100)
    print('-------------------------')

def runFeatureSelection(X,y):
    # Inizializzazione DEAP
    creator.create("FitnessMax", base.Fitness, weights=(-1.0, -1.0, -1.0))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", np.random.choice, [False, True])
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, len(X.columns))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluate, X=X, y=y)  # Utilizzo di X e y senza suddivisione
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Esecuzione dell'algoritmo genetico
    GeneticAlgorithm.runGA(toolbox, X)