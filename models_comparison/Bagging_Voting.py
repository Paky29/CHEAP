import pandas as pd
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, VotingRegressor
from sklearn.neural_network import MLPRegressor
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings

class EffortEstimationSoftwareDevelopment:
    def __init__(self):
        self.cols = ['Organization type', 'Size of IT department',
       'Customer organization type', 'Estimated  duration', 'Development type',
       'Application domain', 'Object points', 'actual', 'Contract maturity',
       'Government policy impact', 'Organization management structure clarity',
       'Developer hiring policy', 'Developer incentives policy ',
       'Developer training', 'Development team management',
       'Top management support', 'Top management opinion of previous system',
       'User resistance', 'User computer experience', ' Users stability ',
       ' Requirements flexibility ', 'Project manager experience',
       'Consultant availability', 'DBMS  expert availability',
       'Precedentedness', 'Software tool experience',
       'Programmers experience in programming language',
       'Programmers capability ', 'Team selection', 'Team size',
       'Daily working hours', 'Team contracts', 'Team cohesion',
       'Schedule quality', 'Development environment adequacy',
       'Tool availability ', 'Methodology', 'Programming language used',
       'DBMS used', 'Technical stability', 'Open source software',
       'Degree of software reuse ', 'Degree of risk management',
       'Use of standards', ' Process reengineering ',
       'Requirement accuracy level', 'Technical documentation', 'User manual',
       'Required reusability', 'Performance requirements',
       'Product complexity', 'Security requirements',
       'Reliability requirements', 'Specified H/W']
        self.df = pd.read_csv('../datasets/seera_nofs.csv', names=self.cols)
        self.covarianceMatrix = None
        self.eigenVals = None
        self.eigenVecs = None
        self.pcaDataset = self.df.drop('actual', axis=1)
        self.folds = {}
        self.stats = {}
        self.estimators = {}
        self.splitDataIntoFolds()
        self.runBaggingKNN()
        self.formEstimatorList()
        self.runVotingEnsemble()

    def splitDataIntoFolds(self):
        for i in range(10):
            self.folds[i] = {
                'train': [],
                'test': []
            }
        kf = KFold(n_splits=10, shuffle=True, random_state=0)
        counter = 0
        for train_index, test_index in kf.split(self.pcaDataset):
            self.folds[counter % 10]['train'].append(train_index)
            self.folds[counter % 10]['test'].append(test_index)
            counter += 1

    def trainTestSplit(self, counter):
        trainX = list(self.pcaDataset.values[self.folds[counter]['train']][0])
        trainY = list(self.df['actual'].values[self.folds[counter]['train']][0])
        testX = list(self.pcaDataset.values[self.folds[counter]['test']][0])
        testY = list(self.df['actual'].values[self.folds[counter]['test']][0])
        return (trainX, trainY, testX, testY)


    def runBaggingKNN(self):
        print('KNN Bagging Regression Stats:')
        for k in range(1, 10):
            predictions = []
            actual = []
            testCounter = 0
            while testCounter < 10:
                trainX, trainY, testX, testY = self.trainTestSplit(testCounter)
                regressor = BaggingRegressor(estimator=KNeighborsRegressor(
                        n_neighbors=k), n_estimators=2000, random_state=0, n_jobs=-1, max_samples=55,
                        bootstrap_features=True, oob_score=True)
                regressor.fit(trainX, trainY)
                predictions += list(regressor.predict(testX))
                actual += testY
                testCounter += 1
            print('\nFor k = ', k)
            print('RMSE: ', np.sqrt(mean_squared_error(actual, predictions)))
            print('Mean Absolute error: ', mean_absolute_error(actual, predictions))
            print('MMRE: ', np.mean(np.abs(np.array(actual) - np.array(predictions)) / np.array(actual)))
            print('R2 score: ', r2_score(actual, predictions))

    def formEstimatorList(self):
        self.estimators = [
            ('linear', LinearRegression()),
            ('ridge', RidgeCV(cv=10, alphas=[1e-3, 1e-2, 1e-1, 1])),
            ('lasso', LassoCV(cv=10, random_state=0)),
            ('elasticNet', ElasticNetCV(cv=10, random_state=0)),
            ('mlp', MLPRegressor(random_state=0, max_iter=10)),
            ('randomForest', RandomForestRegressor(
                n_estimators=2000, random_state=0, max_features=0.3)),
            ('knn', KNeighborsRegressor(n_neighbors=2))
        ]

    @ignore_warnings(category=ConvergenceWarning)
    def runVotingEnsemble(self):
        print('\nVoting Regression Stats:\n'.format(type))
        predictions = []
        actual = []
        testCounter = 0
        while testCounter < 10:
            trainX, trainY, testX, testY = self.trainTestSplit(testCounter)
            regressor = VotingRegressor(estimators=self.estimators)
            regressor.fit(trainX, trainY)
            predictions += list(regressor.predict(testX))
            actual += testY
            testCounter += 1

        print('RMSE: ', np.sqrt(mean_squared_error(actual, predictions)))
        print('Mean Absolute error: ', mean_absolute_error(actual, predictions))
        print('MMRE: ', np.mean(np.abs(np.array(actual) - np.array(predictions)) / np.array(actual)))
        print('R2 score: ', r2_score(actual, predictions))

if __name__ == '__main__':
    EffortEstimationSoftwareDevelopment()
