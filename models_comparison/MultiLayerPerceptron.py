import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

def run():
    # Set the parameters
    convergence_tol = 1e-5
    max_iterations = 10000
    min_gradient = 1e-6

    # Split the data into training and validation sets
    seera = pd.read_csv("datasets/SEERA_train.csv", delimiter=',', decimal=".")
    X = seera.drop(['Actual effort'], axis=1)
    y = seera['Actual effort']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23)

    # Create a 10-fold cross-validation object
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    mlp_store = []

    for num_neurons in range(1, 11):  # Adjust the range as per your requirement
        variances = []
        train_errors = []

        for train_index, val_index in kf.split(X_train):
            X_train_fold, X_val_fold = X.loc[X.index[train_index]], X.loc[X.index[val_index]]
            y_train_fold, y_val_fold = y[train_index], y[val_index]

            mlp=MLPRegressor(hidden_layer_sizes=(num_neurons,), tol=convergence_tol,
                       max_iter=max_iterations, alpha=min_gradient)

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_train_fold)

            # Assign column names to the scaled data
            X_scaled = pd.DataFrame(X_scaled, columns=X_train_fold.columns)

            mlp.fit(X_scaled, y_train_fold)

            mlp_store.append(mlp)

            y_train_pred = mlp.predict(X_scaled)
            train_error = mean_squared_error(y_train_fold, y_train_pred)

            y_val_pred = mlp.predict(X_val_fold)
            val_error = mean_squared_error(y_val_fold, y_val_pred)

            variances.append(val_error)
            train_errors.append(train_error)

            # Stop training if both training error decreases and validation error begins to increase
            if len(variances) >= 2 and train_errors[-1] > train_errors[-2] and val_error > variances[-2]:
                break

        avg_variance = np.mean(variances)

        print(f"Hidden Neurons: {num_neurons}, Average Variance: {avg_variance}")

    best_mlp = mlp_store[-2]
    y_pred = best_mlp.predict(X_test)

    print(best_mlp)

    print("RMSE: ", np.sqrt(mean_squared_error(y_test, y_pred)))
    print("R2: ", r2_score(y_test, y_pred))
    print("mre: ", np.mean(np.abs(y_test - y_pred) / y_test) * 100)
