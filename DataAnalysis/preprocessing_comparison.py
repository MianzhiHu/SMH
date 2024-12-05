import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
folder_path = './Data/'
all_results = {}

for file in os.listdir(folder_path):
    if file.endswith('.csv') and 'modeled' not in file:
        data = pd.read_csv(folder_path + file)
        file_name = file.split('_')[-2] + '_' + file.split('_')[-1].replace('.csv', '')
        all_results[file_name] = data

# Calculate the correlation for estimated anticipatory GSR versus each other files
target = ['PhasicAnticipatoryGSRAUC', 'PhasicOutcomeGSRAUC', 'TonicAnticipatoryGSRAUC', 'TonicOutcomeGSRAUC']

# Threshold for acceptable NaN proportion
threshold = 0.9

# Iterate over each variable in the target list
for col in target:
    # Create a DataFrame to store the specified column from all files
    combined_columns = pd.DataFrame()

    # Iterate through all DataFrames in the dictionary and extract the target column
    for filename, df in all_results.items():
        if col in df.columns:
            # Add the target column from the current file to the combined DataFrame
            combined_columns[filename] = df[col]

    # Check if columns have too many missing values, and remove those columns
    filtered_combined_columns = combined_columns.loc[:, combined_columns.isna().mean() <= threshold]

    # Remove rows with any remaining missing values
    filtered_combined_columns = filtered_combined_columns.dropna()

    # # log-transform the data
    # filtered_combined_columns = np.log(filtered_combined_columns + 1)

    # Calculate the correlation matrix of the filtered combined columns
    correlation = filtered_combined_columns.corr()

    # Plot the correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, xticklabels='auto', yticklabels='auto', cmap='coolwarm')
    plt.title(f"Correlation Matrix for {col}")
    plt.tight_layout()

    # Save the plot with a unique filename based on the current target column
    plt.savefig(f'./figures/correlation_matrix_{col}.png')
    plt.clf()