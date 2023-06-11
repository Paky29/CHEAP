import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt

# determine the values NaN with regression
def defineNaN(seera):
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

#median-MAD scaling
def scalingMAD(seera):
    effort = seera['Actual effort']
    seera = seera.drop(['Actual effort'],axis=1)
    x_columns = seera.columns
    # Create an instance of the RobustScaler
    scaler = RobustScaler()

    # Fit the scaler to the data and transform it
    scaled_data = scaler.fit_transform(seera)

    seera = pd.DataFrame(scaled_data, columns=x_columns)
    seera['Effort'] = effort

def missingValues(seera):
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
def saveDataset(dataset,name):
    dataset.to_csv("../datasets/"+name, index=False)

def cleanDataset():
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

    defineNaN(seera)
    scalingMAD(seera)

    saveDataset(seera,"SEERA_cleaned.csv")
    return seera
