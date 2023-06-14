import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# determine the values NaN with regression
def define_NaN(seera):
    # Select the columns with missing values
    missing_cols = seera.columns[seera.isnull().any()]

    # Perform regression imputation for each column with missing values
    for col in missing_cols:
        # Select the complete columns for the regression
        complete_cols = list(set(seera.columns) - set(missing_cols) - set([col]))

        # Split the data into complete and missing rows for the column
        complete_rows = seera.dropna(subset=[col])
        missing_rows = seera[col].isnull()

        # Fit a linear regression model
        model = DecisionTreeRegressor()
        model.fit(complete_rows[complete_cols], complete_rows[col])

        # Replace missing values with predicted values
        seera.loc[missing_rows, col] = model.predict(seera[complete_cols])[missing_rows]
        return seera

#median-MAD scaling
def scaling_robust(seera):
    effort = seera['Actual effort']
    seera = seera.drop(['Actual effort'],axis=1)
    x_columns = seera.columns
    # Create an instance of the RobustScaler
    scaler = RobustScaler()

    # Fit the scaler to the data and transform it
    scaled_data = scaler.fit_transform(seera)

    seera = pd.DataFrame(scaled_data, columns=x_columns)
    seera['Effort'] = effort
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
    dataset.to_csv("datasets/"+name, index=False)

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

    #replace '?' value with NaN value
    old_value = '?'
    new_value = np.nan
    seera.replace(old_value, new_value, inplace=True)

    #missingValues(seera)

    #drop row 69, too many NaN values
    seera = seera.drop(69)
    seera = seera.reset_index(drop=True)

    define_NaN(seera)

    #0.2 because there are not many outliers from our plot
    remove_outliers(seera, 0.2)

    # Change values for different type of user manual in one value for the presence of the user manual
    seera.loc[seera['User manual'] != 1, 'User manual'] = 2

    scaling_MAD(seera)

    save_dataset(seera,"SEERA_cleaned.csv")
    return seera
