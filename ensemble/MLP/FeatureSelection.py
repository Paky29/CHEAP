import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import KFold
from deap import base, creator, tools
import time
import datetime
import pickle
import concurrent.futures
from ensemble.MLP.MLP_MetaModel import Ensemble

def runFeatureSelection(model):
    creator.create("FitnessMax", base.Fitness, weights=(-1.0, -1.0, -1.0))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", np.random.choice, [False, True])
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, len(model.X.columns))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate, model=model)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    runGA(toolbox, model.X)

def evaluate_fold(train_indices, test_indices,X_selected, model):
        X_train = X_selected.iloc[train_indices]
        X_test = X_selected.iloc[test_indices]
        y_train = model.y.iloc[train_indices]
        y_test = model.y.iloc[test_indices]

        rf_regressor = RandomForestRegressor()
        svr = SVR(kernel='rbf')
        elasticnet_regressor = ElasticNet()
        knn_regressor = KNeighborsRegressor()
        gb_regressor = GradientBoostingRegressor()

        # Define weak learners
        weak_learners = [('rf', rf_regressor),
                         ('knn', knn_regressor),
                         ('svr', svr),
                         ('gb', gb_regressor),
                         ('en', elasticnet_regressor)]

        # Final learner or metamodel
        final_learner = MLPRegressor(random_state=1, max_iter=500)

        train_meta_model = None
        test_meta_model = None

        # Start stacking
        for clf_id, clf in weak_learners:
            # Predictions for each classifier based on k-fold
            predictions_clf = model.k_fold_cross_validation(clf)

            # Predictions for test set for each classifier based on train of level 0
            test_predictions_clf = model.train_level_0(clf)

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
        model.train_level_1(final_learner, train_meta_model, test_meta_model)

        predictions = final_learner.predict(test_meta_model)

        # Calcolo delle misure di performance
        rmse = np.sqrt(mean_squared_error(model.y_test, predictions))
        r2 = r2_score(model.y_test, predictions)
        mre = np.mean(np.abs(model.y_test - predictions) / model.y_test) * 100

        return rmse,r2, mre

def evaluate(individual, model):
    selected_features = [feature for feature, mask in zip(model.X.columns, individual) if mask]
    X_selected = model.X[selected_features]

    kfold = KFold(n_splits=10, shuffle=True, random_state=42)

    rmse_scores = []
    r2_scores = []
    mre_scores = []

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []
        for train_indices, test_indices in kfold.split(X_selected):
            future = executor.submit(evaluate_fold, train_indices, test_indices, X_selected, model)
            futures.append(future)

        for future in concurrent.futures.as_completed(futures):
            rmse, r2, mre = future.result()
            rmse_scores.append(rmse)
            r2_scores.append(r2)
            mre_scores.append(mre)

    mean_rmse = np.mean(rmse_scores)
    mean_r2 = np.mean(r2_scores)
    mean_mre = np.mean(mre_scores)

    return mean_rmse, mean_r2, mean_mre

def runGA(toolbox, X):
    population_size = 7
    generations = 3
    pop = toolbox.population(n=population_size)

    for gen in range(generations):
        print(f"\nGeneration {gen + 1}")

        # Valutazione della popolazione
        fitnesses = toolbox.map(toolbox.evaluate, pop)
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        # Stampa delle metriche della popolazione attuale
        rmse_values = [ind.fitness.values[0] for ind in pop]
        r2_values = [ind.fitness.values[1] for ind in pop]
        mre_values = [ind.fitness.values[2] for ind in pop]
        print(f"RMSE values: {rmse_values}")
        print(f"R^2 values: {r2_values}")
        print(f"MRE values: {mre_values}")

        # Stampa della popolazione attuale
        print(f"\nPopulation of Generation {gen + 1}:")
        for i, ind in enumerate(pop):
            print(f"Individual {i + 1}: {ind}")

            # Selezione degli individui migliori
        offspring = toolbox.select(pop, len(pop))

        # Clonazione degli individui selezionati
        offspring = list(map(toolbox.clone, offspring))

        # Applicazione degli operatori di crossover e mutazione
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if np.random.random() < 0.5:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if np.random.random() < 0.2:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Valutazione degli individui appena generati
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Sostituzione della popolazione con la nuova generazione
        pop[:] = offspring

    # Selezione dell'individuo migliore
    best_individual = tools.selBest(pop, k=1)[0]
    selected_features = [feature for feature, mask in zip(X.columns, best_individual) if mask]

    # Best Individual fitness metrics
    print("\n--Selected features--")
    print(selected_features)
    print("--------------------------")
    print("--Best individual fitness--")
    print("RMSE:", best_individual.fitness.values[0])
    print("R^2 score:", best_individual.fitness.values[1])
    print("MRE:", best_individual.fitness.values[2])
    print("--------------------------")

def main():
    # Start
    start_time = time.time()
    model = Ensemble()

    # Serializzazione dell'oggetto Ensemble
    with open('ensemble.pkl', 'wb') as file:
        pickle.dump(model, file)

    # Deserializzazione dell'oggetto Ensemble
    with open('ensemble.pkl', 'rb') as file:
        ensemble_deserialized = pickle.load(file)

    ensemble_deserialized.load_data()
    runFeatureSelection(ensemble_deserialized)

    # End
    print("Execution time:", str(datetime.timedelta(seconds=time.time() - start_time)))

if __name__ == "__main__":
    main()