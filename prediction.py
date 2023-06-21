import os

import pandas as pd
from joblib import load
from data_cleaning.cleanSeera import scaling_robust, save_dataset
from ensemble import LinearRegression_MetaModel3
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import numpy as np

app = FastAPI()

class Risultati(BaseModel):
    index: float
    prediction: float

class PredictionRequest(BaseModel):
    input_data: list

class RealEffort(BaseModel):
    input_data: list

@app.post("/predict")
def predict_data(request: PredictionRequest):
    input_data = request.input_data

    seera = pd.read_csv("datasets/SEERA_retrain.csv", delimiter=',', decimal=".")
    last_row_index = seera['Indice Progetto'].max() + 1
    input_data.insert(0,last_row_index)
    input_data.append("NaN")

    seera = seera.append(pd.Series(input_data, index=seera.columns), ignore_index=True)

    save_dataset(seera, 'SEERA_retrain.csv')

    seera = seera.drop(['Indice Progetto','Actual effort'], axis=1)
    seera = scaling_robust(seera)

    input_scaled = seera.tail(1)

    rf_regressor = load("models_saved/random_forest.joblib")
    svr = load("models_saved/svr.joblib")
    gb_regressor = load("models_saved/gradient_boosting.joblib")
    knn_regressor = load("models_saved/knn.joblib")
    elasticnet_regressor = load("models_saved/elastic_net.joblib")

    y_pred_rf = rf_regressor.predict(input_scaled)
    y_pred_svr = svr.predict(input_scaled)
    y_pred_gb = gb_regressor.predict(input_scaled)
    y_pred_knn = knn_regressor.predict(input_scaled)
    y_pred_elastic = elasticnet_regressor.predict(input_scaled)

    X_meta = np.column_stack((y_pred_gb, y_pred_rf, y_pred_svr, y_pred_knn, y_pred_elastic))
    meta_regressor = load('models_saved/meta_regressor.joblib')
    prediction = meta_regressor.predict(X_meta)
    print(prediction)
    risultato = Risultati(index=last_row_index,prediction=prediction)
    return risultato

@app.post("/update")
def retraining(request:RealEffort):
    index = request.input_data[0]
    real_effort = request.input_data[1]

    seera = pd.read_csv("datasets/SEERA_retrain.csv", delimiter=',', decimal=".")

    # Retrieve the last row of the dataset
    row = seera.loc[seera['Indice Progetto']==index]

    row['Actual effort'] = real_effort

    # Update the modified last row in the dataset
    seera.loc[seera['Indice Progetto']==index] = row

    save_dataset(seera, "SEERA_retrain.csv")

    #seera.drop('Indice Progetto', axis=1)
    seera.dropna(subset=['Actual effort'])
    seera = scaling_robust(seera)

    save_dataset(seera,'SEERA_train.csv')
    seera = seera.drop('Indice Progetto',axis =1)
    LinearRegression_MetaModel3.run(seera.drop('Actual effort', axis=1), seera['Actual effort'])

# Avvia il server FastAPI
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)


