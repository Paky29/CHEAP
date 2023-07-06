import os
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# determine the values NaN with Knn
def define_NaN(seera):
    # Crea un oggetto KNNImputer
    imputer = KNNImputer(n_neighbors=2)

    # Applica l'imputazione ai dati
    X_imputed = imputer.fit_transform(seera)

    # Trasforma l'array NumPy in un DataFrame mantenendo i nomi delle colonne
    seera = pd.DataFrame(X_imputed, columns=seera.columns)

    return seera

#calculate e view the missing values in the dataset
def missing_values(seera):
    # set style graph
    plt.style.use("ggplot")

    # set limit for display
    pd.set_option('display.max_columns', 1000)
    pd.set_option('display.max_rows', 1000)

    rows_with_missing_values = seera[seera.isnull().any(axis=1)]
    print(rows_with_missing_values)

    # Visualize rows with missing values
    plt.matshow(rows_with_missing_values.isnull())
    plt.xlabel('Column')
    plt.ylabel('Row')
    plt.show()

# Save the modified data to a new CSV file
def save_dataset(dataset,name):
    filename = "datasets/"+name
    if os.path.exists(filename):
        os.remove(filename)
    dataset.to_csv(filename, index=False)

#delete outlier
def remove_outliers(data, contamination):
    # Create an Isolation Forest model
    model = IsolationForest(contamination=contamination)

    # Fit the model and predict outliers
    outliers = model.fit_predict(data)

    # Filter out the outliers from the dataset
    filtered_data = data[outliers != -1]
    filtered_data = filtered_data.reset_index(drop=True)

    return filtered_data

def feature_selection(seera):
    seera = seera[['Customer organization type', 'Estimated  duration', 'Application domain',
          'Developer training', 'Development team management','Top management support',' Requirements flexibility ',
          'Consultant availability', 'DBMS  expert availability','Software tool experience', 'Team size', 'Development environment adequacy',
          'Tool availability ', 'DBMS used','Requirement accuracy level', 'Technical documentation', 'Required reusability',
          'Performance requirements','Actual effort']]
    return seera

def pearson_correlation(seera):
    correlation_matrix = seera.corr(method='pearson')

    # Display the correlation matrix
    print(correlation_matrix)

def clean_dataset():
    #load original dataset
    seera = pd.read_csv("../datasets/SEERA.csv", delimiter=';', decimal=",")

    #change decimal from ',' to '.'
    seera = seera.applymap(lambda x: x.replace(',', '.') if isinstance(x, str) else x)
    seera = seera.apply(pd.to_numeric, errors='coerce', downcast='float')

    #delete useless feature or calculable features at the end of the project
    #high correlation between dedicated member and team size, size of organization and size IT department
    #estimated size e degree of standards usage have too many NaN values
    seera = seera.drop(['Year of project','ProjID','Organization id','Role in organization','Actual duration','Size of organization','% project gain (loss)','Other sizing method','Economic instability impact','Clarity of manual system',' Requirment stability ','Team continuity ','Income satisfaction','# Multiple programing languages ','Level of outsourcing','Outsourcing impact','Comments within the code','Estimated size','Dedicated team members','Degree of standards usage'], axis=1)

    # alta correlazione di Pearson con Team size e Program capability
    seera = seera.drop(['Estimated effort', 'Analysts capability '], axis=1)

    #replace '?' value with NaN value
    old_value = '?'
    new_value = np.nan
    seera.replace(old_value, new_value, inplace=True)

    #drop row 69, too many NaN values
    seera = seera.drop(69)
    seera = seera.reset_index(drop=True)

    seera = define_NaN(seera)

    #0.1 because there are not many outliers from our plot
    remove_outliers(seera, 0.1)

    # Change values for different type of user manual in one value for the presence of the user manual
    seera.loc[seera['User manual'] != 1, 'User manual'] = 2

    feature_selection(seera)

    seera.insert(0, 'Indice Progetto', seera.index)

    save_dataset(seera,"SEERA_train.csv")
    return seera
