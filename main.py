import pandas as pd
from ensemble import LinearRegression_MetaModel, SVR_MetaModel, LinearRegression_MetaModel2, LinearRegression_MetaModel3
from models import ElasticNet, GradientBoostingRegression, KNearestNeighbors, RandomForestRegressor, \
    SupportVectorRegression
from tuning.OptunaTuning import optuna_tuning
from tuning.PSOTuning import pso_tuning


def main():
    seera = pd.read_csv("datasets/SEERA_cleaned.csv", delimiter=',', decimal=".")
    X = seera.drop('Effort', axis=1)
    y = seera['Effort']

    #ElasticNet.run(X, y)
    #GradientBoostingRegression.run(X, y)
    #KNearestNeighbors.run(X, y)
    #RandomForestRegressor.run(X, y)
    #SupportVectorRegression.run(X, y)

    #SVR_MetaModel.run(X, y)

    #LinearRegression_MetaModel.run(X, y)
    #LinearRegression_MetaModel2.run(X, y)
    #LinearRegression_MetaModel3.run(X, y)
    print(pso_tuning())

if __name__=="__main__":
    main()