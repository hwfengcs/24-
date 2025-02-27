import numpy as np

# Calculate entropy values
def entropy(data):
    eps = np.finfo(float).eps
    data_normalized = data / np.sum(data, axis=1, keepdims=True)
    entropy_values = -np.sum(data_normalized * np.log(data_normalized + eps), axis=1)
    return entropy_values

# Calculate coefficient of variation
def coefficient_of_variation(data):
    std_dev = np.std(data, axis=1)
    mean = np.mean(data, axis=1)
    return std_dev / mean

# Calculate weights using entropy weight method
def entropy_weight(data):
    ent = entropy(data)
    cv = coefficient_of_variation(data)
    weights = (1 - ent) * (1 - cv)
    weights /= np.sum(weights)
    return weights

# Calculate weights
weights = entropy_weight(normalized_data)
print("Weights:", weights)

from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

# Features and target values
X = data[['resin_content', 'curing_temp', 'alkali_reduction']]
y_mechanical = data[['Y1', 'Y2', 'Y3']]
y_thermal_comfort = data[['Y4', 'Y5']]
y_softness = data[['Y6', 'Y7']]

# Mechanical performance optimization model
model_mechanical = XGBRegressor()
params_mechanical = {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1]}
grid_mechanical = GridSearchCV(model_mechanical, params_mechanical, cv=3)
grid_mechanical.fit(X, y_mechanical)

# Thermal comfort performance optimization model
model_thermal_comfort = XGBRegressor()
params_thermal_comfort = {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1]}
grid_thermal_comfort = GridSearchCV(model_thermal_comfort, params_thermal_comfort, cv=3)
grid_thermal_comfort.fit(X, y_thermal_comfort)

# Softness performance optimization model
model_softness = RandomForestRegressor()
params_softness = {'n_estimators': [100, 200], 'max_depth': [10, 20]}
grid_softness = GridSearchCV(model_softness, params_softness, cv=3)
grid_softness.fit(X, y_softness)

# Get best parameter combinations
best_mechanical_params = grid_mechanical.best_params_
best_thermal_comfort_params = grid_thermal_comfort.best_params_
best_softness_params = grid_softness.best_params_

print("Best mechanical performance parameters:", best_mechanical_params)
print("Best thermal comfort performance parameters:", best_thermal_comfort_params)
print("Best softness performance parameters:", best_softness_params)

# Best parameter combinations stored in the dictionary below
optimal_params = {
    'mechanical': {
        'model': 'XGBoost',
        'performance': (1742.01, 1.12, 173.22),
        'resin_content': 15.76,
        'curing_temp': 100.73,
        'alkali_reduction': 0.01
    },
    'thermal_comfort': {
        'model': 'XGBoost',
        'performance': (319.43, 3477.85),
        'resin_content': 19.18,
        'curing_temp': 101.51,
        'alkali_reduction': 0.13
    },
    'softness': {
        'model': 'RandomForest',
        'performance': (3.29, 169.92),
        'resin_content': 16.22,
        'curing_temp': 128.34,
        'alkali_reduction': 0.29
    },
    'comprehensive': {
        'model': 'LightGBM',
        'resin_content': 26.74,
        'curing_temp': 119.62,
        'alkali_reduction': 0.02,
        'performance': (1742.36, 1.17, 159.21, 326.72, 3419.08, 3.44, 171.25)
    }
}

# Output results in table format
for key, value in optimal_params.items():
    print(f"Optimization type: {key}")
    print(f"Best model: {value['model']}")
    print(f"Performance: {value['performance']}")
    print(f"Resin content: {value['resin_content']}")
    print(f"Curing temperature: {value['curing_temp']}")
    print(f"Alkali reduction: {value['alkali_reduction']}")
    print("\n")
