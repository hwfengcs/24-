# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from pyswarm import pso  # PSO implementation library

# Load your dataset
data = pd.read_csv('data.csv')

# Separate features (process parameters) and targets (performance indicators)
X = data[['x1', 'x2', 'x3']]  # Replace with actual column names
performance_indicators = ['y1', 'y2', 'y3', 'y4', 'y5', 'y6', 'y7']

# Split data into train, validation, and test sets (7:3 ratio)
X_train, X_valtest, y_train_all, y_valtest_all = train_test_split(X, data[performance_indicators], test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_valtest, y_valtest_all, test_size=0.33, random_state=42)

# Define regression models to be evaluated
models = {
    'Linear Regression': LinearRegression(),
    'Naive Bayes': GaussianNB(),
    'Support Vector Machine': SVR(),
    'Multi-layer Perceptron': MLPRegressor(random_state=42),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42),
    'AdaBoost': AdaBoostRegressor(random_state=42),
    'XGBoost': XGBRegressor(random_state=42),
    'CatBoost': CatBoostRegressor(random_state=42, verbose=0),
    'LightGBM': LGBMRegressor(random_state=42)
}

# Define evaluation function
def evaluate_model(model, X_train, y_train, X_val, y_val):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    return mse, rmse, mae, r2

# Evaluate all models for each performance indicator
results = {}
for pi in performance_indicators:
    y_train = y_train_all[pi]
    y_val_ = y_val[pi]
    results[pi] = {}
    for name, model in models.items():
        mse, rmse, mae, r2 = evaluate_model(model, X_train, y_train, X_val, y_val_)
        results[pi][name] = {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2}
        print(f"Model: {name}, Performance Indicator: {pi}, MSE: {mse}, RMSE: {rmse}, MAE: {mae}, R2: {r2}")

# Determine the best model for each performance indicator based on R2
best_models = {pi: max(results[pi], key=lambda x: results[pi][x]['R2']) for pi in performance_indicators}

print("Best models for each performance indicator:", best_models)


# Example PSO function for optimizing parameters for a given performance indicator using the best model
def objective_function(params, model, scaler):
    params = np.array(params).reshape(1, -1)
    params = scaler.transform(params)  # Ensure the parameters are scaled if using scaling
    predicted_y = model.predict(params)
    return -predicted_y  # Negative because PSO minimizes, and we want to maximize the performance indicator

# Define parameter bounds based on your process ranges
# Replace min1, max1, min2, max2, etc., with actual parameter bounds
bounds = [(min1, max1), (min2, max2), (min3, max3)]

# Run PSO optimization for each performance indicator
optimized_parameters = {}
for pi in performance_indicators:
    best_model_name = best_models[pi]
    best_model = models[best_model_name]
    best_model.fit(X_train, y_train_all[pi])  # Ensure the model is trained with the best model
    scaler = StandardScaler().fit(X_train)  # Fit scaler on training data if needed

    best_params, _ = pso(objective_function, lb=[bound[0] for bound in bounds], ub=[bound[1] for bound in bounds],
                         args=(best_model, scaler))
    optimized_parameters[pi] = best_params

print("Optimized Parameters for each performance indicator:", optimized_parameters)


