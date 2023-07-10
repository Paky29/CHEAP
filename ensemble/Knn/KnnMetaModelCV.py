from sklearn.ensemble import StackingRegressor
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
import numpy as np
import smogn
from joblib import load
from joblib import dump

def data_balancing(X, y):
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


def runCV():
    seera = pd.read_csv("datasets/SEERA_train.csv", delimiter=',', decimal=".")
    y = seera['Actual effort']
    X = seera.drop(['Actual effort','Indice Progetto'],axis = 1)
    rf_regressor = RandomForestRegressor(max_depth=100, max_features=1.0, min_samples_split=4, n_estimators=49)
    ada = AdaBoostRegressor(learning_rate=0.04371275290084875, loss='square',n_estimators=74)
    elastic = ElasticNet(alpha=0.993873255789175, l1_ratio=0.2679522850857919,selection='random', tol=0.0009877700402121745)
    svr = SVR(C=4.890977754867943, epsilon=0.2963457122766287, gamma=0.05229363983979079,kernel='linear')
    #xgb = xgb.XGBRegressor(colsample_bytree=0.41470186669055753, gamma=0.06493204581815717, learning_rate=0.028343176856406638, max_depth=3, n_estimators=630, min_child_weight=1, subsample = 0.876091138265439, reg_lambda = 0.40025015210995907)


    # Define weak learners
    estimators = [
        ('rf', rf_regressor),
        ('ada', ada),
        ('svr', svr),
        ('en', elastic)
    ]

    reg = StackingRegressor(estimators=estimators,
                            cv = 7,
                            final_estimator=make_pipeline(MinMaxScaler(feature_range=(-1,1)),KNeighborsRegressor(n_neighbors = 5, leaf_size = 30, weights = 'distance', p = 1)))
    #8
    kfold = KFold(n_splits=8, shuffle=True, random_state=23)

    # Liste per memorizzare le performance dei modelli in ogni fold
    rmse_scores = []
    r2_scores = []
    mre_scores = []
    pred_scores = []
    mmre_scores = []

    # Suddivisione dei dati utilizzando la k-fold cross-validation
    for train_indices, test_indices in kfold.split(X):
        X_train = X.iloc[train_indices]
        X_test = X.iloc[test_indices]
        y_train = y.iloc[train_indices]
        y_test = y.iloc[test_indices]

        X_train, y_train = data_balancing(X_train, y_train)
        reg.fit(X_train, y_train)

        predictions = reg.predict(X_test)

        print(predictions)
        print(y_test)

        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)
        mre = np.abs(y_test - predictions) / y_test
        p = mre[mre<.25]

        rmse_scores.append(rmse)
        r2_scores.append(r2)
        mre_scores.append(np.mean(mre))
        pred_scores.append((p.size/mre.size)*100)
        #mmre_scores.append(np.mean(mre))

    mean_rmse = np.mean(rmse_scores)
    mean_r2 = np.mean(r2_scores)
    mean_mre = np.mean(mre_scores)
    mean_pred = np.mean(pred_scores)

    # Stampa delle performance medie
    print('Average RMSE:', mean_rmse)
    print('Average R^2 score:', mean_r2)
    print('Average MRE:', mean_mre)
    print('Average Pred:', mean_pred)
    print('-------------------------')

def run():
    seera = pd.read_csv("datasets/SEERA_train.csv", delimiter=',', decimal=".")
    y = seera['Actual effort']
    X = seera.drop(['Actual effort', 'Indice Progetto'], axis=1)
    rf_regressor = RandomForestRegressor(max_depth=100, max_features=1.0, min_samples_split=4, n_estimators=49)
    ada = AdaBoostRegressor(learning_rate=0.04371275290084875, loss='square', n_estimators=74)
    elastic = ElasticNet(alpha=0.993873255789175, l1_ratio=0.2679522850857919, selection='random',
                         tol=0.0009877700402121745)
    svr = SVR(C=4.890977754867943, epsilon=0.2963457122766287, gamma=0.05229363983979079, kernel='linear')

    # Define weak learners
    estimators = [
        ('rf', rf_regressor),
        ('ada', ada),
        ('svr', svr),
        ('en', elastic)
    ]

    meta = make_pipeline(MinMaxScaler(feature_range=(-1, 1)),KNeighborsRegressor(n_neighbors=5, leaf_size=30,weights='distance', p=1))
    reg = StackingRegressor(estimators=estimators, cv=7, final_estimator=meta)

    X, y = data_balancing(X, y)
    reg.fit(X, y)

    dump(reg, "models_saved/meta_regressor.joblib")

def train():
    seera = pd.read_csv("datasets/SEERA_train.csv", delimiter=',', decimal=".")
    y = seera['Actual effort']
    X = seera.drop(['Actual effort','Indice Progetto'],axis = 1)
    meta_regressor = load("models_saved/meta_regressor.joblib")
    X, y = data_balancing(X, y)
    meta_regressor.fit(X,y)
    dump(meta_regressor, "models_saved/meta_regressor.joblib")
