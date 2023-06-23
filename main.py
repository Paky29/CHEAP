import os

import pandas as pd

from bot.command import run_bot
from data_cleaning.cleanSeera import save_dataset, scaling_robust
from ensemble import LinearRegression_MetaModel, SVR_MetaModel, LinearRegression_MetaModel2, LinearRegression_MetaModel3
from models import ElasticNet, GradientBoostingRegression, KNearestNeighbors, RandomForestRegressor, \
    SupportVectorRegression


def main():
    #seera = pd.read_csv("datasets/SEERA_train.csv", delimiter=',', decimal=".")
    #X = seera[['Indice Progetto','Organization type', 'Customer organization type', 'Estimated  duration', 'Application domain', 'Government policy impact', 'Organization management structure clarity', 'Developer hiring policy', 'Developer training', 'Development team management', ' Requirements flexibility ', 'Project manager experience', 'DBMS  expert availability', 'Precedentedness', 'Software tool experience', 'Team size', 'Daily working hours', 'Team contracts', 'Schedule quality', 'Degree of risk management', 'Requirement accuracy level', 'User manual', 'Required reusability', 'Product complexity', 'Security requirements', 'Specified H/W', 'Actual effort']]
    #X = X.drop([0,26,40,42,10])
    #save_dataset(X,"SEERA_retrain.csv")
    #seera = scaling_robust(seera)
    #X = seera.drop(['Indice Progetto','Actual effort'],axis=1)
    #y = seera['Actual effort']

    #LinearRegression_MetaModel3.run(X,y)
    #print(pso_tuning())
    run_bot()

if __name__=="__main__":
    main()
