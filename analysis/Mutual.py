import pandas as pd
from sklearn.feature_selection import mutual_info_regression

# row limits = 100
pd.set_option('display.max_rows', 100)

dataset = pd.read_csv("../datasets/SEERA.csv", delimiter=',', decimal=".")
X = dataset.drop('Effort', axis=1)
y = dataset['Effort']


def mutualInformation():
    # Calcolo della mutual information
    mutual_info = mutual_info_regression(X, y)

    # Creazione di un DataFrame per visualizzare i risultati
    results = pd.DataFrame({'Feature': X.columns, 'Mutual Information': mutual_info})

    print(results.to_string(index=False))


def kMutualInformation(k):
    mutual_info = mutual_info_regression(X, y)
    results = pd.DataFrame({'Feature': X.columns, 'Mutual Information': mutual_info})
    results_sorted = results.nlargest(k, 'Mutual Information')
    print(results_sorted)


if __name__ == '__main__':
    scelta = int(input(">mutual or kMutual: (1/0)"))
    if scelta == 1:
        mutualInformation()
    else:
        k = int(input(">insert k:"))
        kMutualInformation(k)
