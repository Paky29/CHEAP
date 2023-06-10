import pandas as pd

from ensemble import LinearRegression_MetaModel, SVR_MetaModel, LinearRegression_MetaModel2, LinearRegression_MetaModel3
from models import ElasticNet, GradientBoostingRegression, KNearestNeighbors, RandomForestRegressor, \
    SupportVectorRegression

seera = pd.read_csv("datasets/SEERA.csv", delimiter=',', decimal=".")
X = seera.drop('Effort', axis=1)
y = seera['Effort']

ElasticNet.run(X, y)
GradientBoostingRegression.run(X, y)
KNearestNeighbors.run(X, y)
RandomForestRegressor.run(X, y)
SupportVectorRegression.run(X, y)

SVR_MetaModel.run(X, y)

LinearRegression_MetaModel.run(X, y)
LinearRegression_MetaModel2.run(X, y)
LinearRegression_MetaModel3.run(X, y)
