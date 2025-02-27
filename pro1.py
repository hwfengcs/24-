import pandas as pd

# Assuming you have 7 separate CSV files to merge
file_paths = ["data1.csv", "data2.csv", "data3.csv", "data4.csv", "data5.csv", "data6.csv", "data7.csv"]

# Merge all files into one DataFrame
merged_data = pd.concat([pd.read_csv(f) for f in file_paths])

# Remove duplicate information
merged_data = merged_data.drop_duplicates()

# Save merged data to CSV
merged_data.to_csv("MergedData.csv", index=False)

import matplotlib.pyplot as plt

# Descriptive statistics
stats_table = merged_data.describe()

# Save descriptive statistics to a CSV file
stats_table.to_csv("DescriptiveStats.csv")

# Boxplot for process parameters and performance indicators
process_params = ['ResinContent', 'CuringTemperature', 'AlkaliReduction']
performance_indicators = ['TensileStrength', 'Elongation', 'TearStrength', 'Permeability', 'MoistureTransmission', 'Softness', 'Flexibility']

plt.figure(figsize=(12, 8))
merged_data[process_params].boxplot()
plt.xticks(rotation=45)
plt.title('Boxplot of Process Parameters')
plt.savefig('ProcessParameters_Boxplot.png')
plt.show()

plt.figure(figsize=(12, 8))
merged_data[performance_indicators].boxplot()
plt.xticks(rotation=45)
plt.title('Boxplot of Performance Indicators')
plt.savefig('PerformanceIndicators_Boxplot.png')
plt.show()


from sklearn.preprocessing import StandardScaler

# Select columns to standardize
columns_to_standardize = performance_indicators

# Initialize the scaler
scaler = StandardScaler()

# Fit and transform the data
normalized_data = merged_data.copy()
normalized_data[columns_to_standardize] = scaler.fit_transform(merged_data[columns_to_standardize])

# Save normalized data to CSV
normalized_data.to_csv("NormalizedData.csv", index=False)


import seaborn as sns

# Plot scatter plots for process parameters vs performance indicators
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

sns.scatterplot(ax=axes[0, 0], x='ResinContent', y='MoistureTransmission', data=merged_data)
axes[0, 0].set_title('Resin Content vs Moisture Transmission')

sns.scatterplot(ax=axes[0, 1], x='CuringTemperature', y='TensileStrength', data=merged_data)
axes[0, 1].set_title('Curing Temperature vs Tensile Strength')

sns.scatterplot(ax=axes[0, 2], x='AlkaliReduction', y='TensileStrength', data=merged_data)
axes[0, 2].set_title('Alkali Reduction vs Tensile Strength')

sns.scatterplot(ax=axes[1, 0], x='ResinContent', y='Permeability', data=merged_data)
axes[1, 0].set_title('Resin Content vs Permeability')

sns.scatterplot(ax=axes[1, 1], x='CuringTemperature', y='Permeability', data=merged_data)
axes[1, 1].set_title('Curing Temperature vs Permeability')

sns.scatterplot(ax=axes[1, 2], x='AlkaliReduction', y='TearStrength', data=merged_data)
axes[1, 2].set_title('Alkali Reduction vs Tear Strength')

plt.tight_layout()
plt.savefig('ScatterPlots_ProcessParameters_vs_PerformanceIndicators.png')
plt.show()


import numpy as np
import seaborn as sns

# Calculate Spearman correlation coefficients
correlation_matrix = merged_data.corr(method='spearman')

# Plot heatmap of Spearman correlation coefficients
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Spearman Correlation Coefficient Heatmap')
plt.savefig('SpearmanCorrelationHeatmap.png')
plt.show()


from sklearn.preprocessing import MinMaxScaler

# Normalize the data using Min-Max scaling
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(merged_data[performance_indicators + process_params])

# Reference sequence: mean of each performance indicator
reference_sequence = np.mean(normalized_data, axis=0)

def grey_relation_coefficient(reference, comparison, resolution_coefficient=0.5):
    diff = np.abs(reference - comparison)
    min_diff = np.min(diff)
    max_diff = np.max(diff)
    return (min_diff + resolution_coefficient * max_diff) / (diff + resolution_coefficient * max_diff)

# Calculate Grey Relational Grades
grey_relation_grades = np.zeros((normalized_data.shape[0], len(performance_indicators)))

for i, indicator in enumerate(performance_indicators):
    for j in range(normalized_data.shape[0]):
        grey_relation_grades[j, i] = grey_relation_coefficient(reference_sequence[i], normalized_data[j, i])

# Calculate mean Grey Relational Grade for each performance indicator
mean_grades = np.mean(grey_relation_grades, axis=0)

# Create a DataFrame to display results
gra_results = pd.DataFrame({
    'Performance Indicator': performance_indicators,
    'Grey Relational Grade': mean_grades
})

# Sort the results by Grey Relational Grade
gra_results = gra_results.sort_values(by='Grey Relational Grade', ascending=False)

# Save the results to a CSV file
gra_results.to_csv("GreyRelationalAnalysisResults.csv", index=False)

# Calculate Spearman correlation coefficients for performance indicators
performance_correlation_matrix = merged_data[performance_indicators].corr(method='spearman')

# Plot heatmap of Spearman correlation coefficients for performance indicators
plt.figure(figsize=(10, 6))
sns.heatmap(performance_correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Spearman Correlation Coefficient Heatmap (Performance Indicators)')
plt.savefig('SpearmanCorrelationHeatmap_PerformanceIndicators.png')
plt.show()

# Normalize the performance indicator data using Min-Max scaling
normalized_performance_data = scaler.fit_transform(merged_data[performance_indicators])

# Reference sequence: mean of each performance indicator
reference_sequence_performance = np.mean(normalized_performance_data, axis=0)

# Calculate Grey Relational Grades for performance indicators
grey_relation_grades_performance = np.zeros((normalized_performance_data.shape[0], len(performance_indicators)))

for i, indicator in enumerate(performance_indicators):
    for j in range(normalized_performance_data.shape[0]):
        grey_relation_grades_performance[j, i] = grey_relation_coefficient(reference_sequence_performance[i], normalized_performance_data[j, i])

# Calculate mean Grey Relational Grade for each performance indicator
mean_grades_performance = np.mean(grey_relation_grades_performance, axis=0)

# Create a DataFrame to display results
gra_performance_results = pd.DataFrame({
    'Performance Indicator': performance_indicators,
    'Grey Relational Grade': mean_grades_performance
})

# Sort the results by Grey Relational Grade
gra_performance_results = gra_performance_results.sort_values(by='Grey Relational Grade', ascending=False)

# Save the results to a CSV file
gra_performance_results.to_csv("GreyRelationalAnalysisResults_PerformanceIndicators.csv", index=False)


import statsmodels.api as sm
from statsmodels.formula.api import ols

# Function to perform stepwise regression
def stepwise_regression(X, y):
    initial_list = []
    included = list(X.columns)
    while True:
        changed = False
        # Forward step
        excluded = list(set(X.columns) - set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included + [new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < 0.05:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed = True
        # Backward step
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max()
        if worst_pval > 0.10:
            changed = True
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
        if not changed:
            break
    return included

# Prepare the data
X = merged_data[process_params]
y = merged_data['TensileStrength']  # Change the target variable as needed

# Add interaction terms
X['X1_X2'] = X['ResinContent'] * X['CuringTemperature']
X['X1_X3'] = X['ResinContent'] * X['AlkaliReduction']
X['X2_X3'] = X['CuringTemperature'] * X['AlkaliReduction']
X['X1_X2_X3'] = X['ResinContent'] * X['CuringTemperature'] * X['AlkaliReduction']

# Perform stepwise regression
included_features = stepwise_regression(X, y)
print("Selected features:", included_features)

# Fit the final model
final_model = sm.OLS(y, sm.add_constant(X[included_features])).fit()
print(final_model.summary())

# Define the model formula
formula = 'TensileStrength ~ C(ResinContent) * C(CuringTemperature) * C(AlkaliReduction)'

# Fit the model
anova_model = ols(formula, data=merged_data).fit()

# Perform ANOVA
anova_results = sm.stats.anova_lm(anova_model, typ=2)
print(anova_results)

# Save ANOVA results to CSV
anova_results.to_csv("ThreeFactorANOVAResults.csv")
