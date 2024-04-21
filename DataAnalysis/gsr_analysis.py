import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
import seaborn as sns
import matplotlib.pyplot as plt
from utilities.utility_processGSR import processGSR, area_under_curve


# load processed data
df = pd.read_csv('./Data/preliminary_data.csv')

# split the data into training and test data
training_data = df[df['Phase'] != 'Test']
test_data = df[df['Phase'] == 'Test']


# decide if the reward should be perceived as a loss
def calculate_cumulative_average(x):
    return x.expanding().mean().shift(1)  # shift(1) to exclude the current trial's reward


# Apply the function to each participant's data
training_data.loc[:, 'Cumulative_Average'] = training_data.groupby('Subnum')['Reward'].transform(calculate_cumulative_average)
training_data.loc[:, 'Loss'] = (training_data['Cumulative_Average'] > training_data['Reward']).astype(int)

# further split the data by condition
baseline_training = training_data[training_data['Condition'] == 'Baseline']
baseline_testing = test_data[test_data['Condition'] == 'Baseline']
print(baseline_training['Subnum'].nunique())

frequency_training = training_data[training_data['Condition'] == 'Frequency']
frequency_testing = test_data[test_data['Condition'] == 'Frequency']
print(frequency_training['Subnum'].nunique())

magnitude_training = training_data[training_data['Condition'] == 'Magnitude']
magnitude_testing = test_data[test_data['Condition'] == 'Magnitude']
print(magnitude_training['Subnum'].nunique())

# test whether people show higher outcome GSR when they choose the bad option
# there doesn't seem to be a significant difference
ttest_ind(training_data[training_data['Loss'] == 0]['OutcomeGSRAUC'], training_data[training_data['Loss'] == 1]['OutcomeGSRAUC'])
print('Outcome GSR for the losses:', training_data[training_data['Loss'] == 1]['OutcomeGSRAUC'].mean())
print('Outcome GSR for the wins:', training_data[training_data['Loss'] == 0]['OutcomeGSRAUC'].mean())

# now analyze anticipatory GSR
for condition in ['Baseline', 'Frequency', 'Magnitude']:
    print(condition)
    data_selected = training_data[training_data['Condition'] == condition]
    for phase in ['Phase 1', 'Phase 2', 'Phase 3', 'Phase 4', 'Phase 5', 'Phase 6']:
        print(phase)
        print(ttest_ind(data_selected[(data_selected['BestOption'] == 0) & (data_selected['Phase'] == phase)]['AnticipatoryGSRAUC'],
                        data_selected[(data_selected['BestOption'] == 1) & (data_selected['Phase'] == phase)]['AnticipatoryGSRAUC']))
        print('Anticipatory GSR for the optimal option:', data_selected[(data_selected['BestOption'] == 1) & (data_selected['Phase'] == phase)]['AnticipatoryGSRAUC'].mean())
        print('Anticipatory GSR for the suboptimal option:', data_selected[(data_selected['BestOption'] == 0) & (data_selected['Phase'] == phase)]['AnticipatoryGSRAUC'].mean())
        print()

for condition in ['Baseline', 'Frequency', 'Magnitude']:
    print(condition)
    data_selected = training_data[training_data['Condition'] == condition]
    for phase in ['Phase 1', 'Phase 2', 'Phase 3', 'Phase 4', 'Phase 5', 'Phase 6']:
        print(phase)
        print(ttest_ind(data_selected[(data_selected['BestOption'] == 0) & (data_selected['Phase'] == phase)]['OutcomeGSRAUC'],
                        data_selected[(data_selected['BestOption'] == 1) & (data_selected['Phase'] == phase)]['OutcomeGSRAUC']))
        print('Outcome GSR for the optimal option:', data_selected[(data_selected['BestOption'] == 1) & (data_selected['Phase'] == phase)]['OutcomeGSRAUC'].mean())
        print('Outcome GSR for the suboptimal option:', data_selected[(data_selected['BestOption'] == 0) & (data_selected['Phase'] == phase)]['OutcomeGSRAUC'].mean())
        print()


# CA trials
for condition in ['Baseline', 'Frequency', 'Magnitude']:
    print(condition)
    data_selected = test_data[(test_data['Condition'] == condition) & (test_data['SetSeen '] == 2)]
    print(ttest_ind(data_selected[data_selected['BestOption'] == 0]['AnticipatoryGSRAUC'],
                    data_selected[data_selected['BestOption'] == 1]['AnticipatoryGSRAUC']))
    print('Anticipatory GSR for C:', data_selected[data_selected['BestOption'] == 1]['AnticipatoryGSRAUC'].mean())
    print('Anticipatory GSR for A:', data_selected[data_selected['BestOption'] == 0]['AnticipatoryGSRAUC'].mean())





