from deap import tools, creator, base
from ensemble.Knn import KnnMetaModelBlend as ensembling
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import time
import datetime
import GeneticAlgorithm

def evaluate(individual, model):
    selected_features = [feature for feature, mask in zip(model.X.columns, individual) if mask]
    X_selected = model.X[selected_features]
    model.X = X_selected

    rf_regressor = RandomForestRegressor(max_depth=67, max_features=1.0, n_estimators=67)
    ada = AdaBoostRegressor(learning_rate=0.05984073241086173, n_estimators=77)
    elastic = ElasticNet(alpha=0.9844970636892896, l1_ratio=0.10523697586301965, selection='random',
                         tol=0.0009844912834425824)
    svr = SVR(C=4.973571744211175, epsilon=0.24236503269448004, gamma=0.06602126901315636, kernel='linear')

    # Define weak learners
    weak_learners = [
        ('rf_regressor', rf_regressor),
        ('ada_regressor', ada),
        ('svr_regressor', svr),
        ('en_regressor', elastic)
    ]

    final_learner = make_pipeline(MinMaxScaler(),
                                  KNeighborsRegressor(n_neighbors=11, leaf_size=22, weights='distance', p=2))

    train_meta_model = None
    test_meta_model = None

    # Start stacking
    for clf_id, clf in weak_learners:
        # Predictions for each classifier based on k-fold
        val_predictions, test_predictions = model.train_level_0(clf_id, clf)

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
    model.train_level_1(final_learner, train_meta_model, test_meta_model)

    return model.rmse, model.r2, model.mre

def runFeatureSelection(model):
    creator.create("FitnessMax", base.Fitness, weights=(-1, -1, -1))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", np.random.choice, [False, True])
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, len(model.X.columns))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate, model=model)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    GeneticAlgorithm.runGA(toolbox, model.X)

def main():
    model = ensembling.Ensemble()
    model.load_data()
    runFeatureSelection(model)

if __name__ == "__main__":
    start_time = time.time()
    main()
    print("Execution time:", str(datetime.timedelta(seconds=time.time() - start_time)))


