import numpy as np
import pandas as pd
import smogn
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score

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

def run():
    seera = pd.read_csv("../datasets/SEERA_train.csv", delimiter=',', decimal=".")
    y = seera['Actual effort']
    X = seera.drop(['Actual effort', 'Indice Progetto'], axis=1)

    ada = AdaBoostRegressor(learning_rate=0.04371275290084875, loss='square', n_estimators=74)
    kfold = KFold(n_splits=8, shuffle=True, random_state=23)

    # Liste per memorizzare le performance dei modelli in ogni fold
    rmse_scores = []
    r2_scores = []
    mre_scores = []
    pred_scores = []

    # Suddivisione dei dati utilizzando la k-fold cross-validation
    for train_indices, test_indices in kfold.split(X):
        X_train = X.iloc[train_indices]
        X_test = X.iloc[test_indices]
        y_train = y.iloc[train_indices]
        y_test = y.iloc[test_indices]

        X_train, y_train = data_balancing(X_train, y_train)
        ada.fit(X_train, y_train)

        predictions = ada.predict(X_test)

        print(predictions)
        print(y_test)

        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)
        mre = np.abs(y_test - predictions) / y_test
        p = mre[mre < .25]

        rmse_scores.append(rmse)
        r2_scores.append(r2)
        mre_scores.append(np.mean(mre))
        pred_scores.append((p.size / mre.size) * 100)

    mean_rmse = np.mean(rmse_scores)
    mean_r2 = np.mean(r2_scores)
    mean_mre = np.mean(mre_scores)
    mean_pred = np.mean(pred_scores)

    # Stampa delle performance medie
    print('Prestazioni AdaBoostingRegression')
    print('Average RMSE:', mean_rmse)
    print('Average R^2 score:', mean_r2)
    print('Average MMRE:', mean_mre)
    print('Average Pred:', mean_pred)
    print('-------------------------')


if __name__ == "__main__":
    run()

