import numpy as np
from scipy.optimize import minimize

# Define weights from AHP and entropy method
weights_criteria = np.array([0.1125, 0.7089, 0.1785])  # Mechanical, Thermal Comfort, Flexibility

weights_subcriteria = np.array([0.278, 0.068, 0.185, 0.135, 0.125, 0.121, 0.088])  # Subcriteria weights

# Function to compute Y12 as a combination of subcriteria
def calculate_Y12(parameters):
    # Assuming parameters include resin content, curing temperature, alkali reduction level
    resin_content, curing_temp, alkali_reduction = parameters
    
    # Example formula for Y12 (adjust based on specific formula derived)
    Y12 = resin_content * weights_subcriteria[0] + curing_temp * weights_subcriteria[1] + alkali_reduction * weights_subcriteria[2]
    + ...  # Include other subcriteria terms similarly
    
    return Y12

# Example objective function to minimize Y12
def objective_function(parameters):
    Y12 = calculate_Y12(parameters)
    return Y12

# Initial guess for parameters (resin content, curing temperature, alkali reduction)
initial_guess = np.array([21.46, 113.15, 0.04])

# Constraints (if any) can be added here based on specific problem requirements

# Perform optimization using Particle Swarm Optimization (PSO) or other suitable method
result = minimize(objective_function, initial_guess, method='Nelder-Mead')  # Example method, choose based on problem specifics

# Extract optimized parameters and Y12 value
optimal_parameters = result.x
optimal_Y12 = calculate_Y12(optimal_parameters)

# Print results
print("Optimized parameters (Resin Content, Curing Temperature, Alkali Reduction):", optimal_parameters)
print("Optimized Y12 value:", optimal_Y12)
