import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
import seaborn as sns
import matplotlib.pyplot as plt
from utilities.utility_processGSR import processGSR, area_under_curve, pairwise_t_test_GSR


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

# now analyze anticipatory and outcome GSR
# there doesn't seem to be a significant difference
pairwise_t_test_GSR(training_data, 'AnticipatoryGSRAUC', 'training')
pairwise_t_test_GSR(training_data, 'OutcomeGSRAUC', 'training')

# CA trials
pairwise_t_test_GSR(test_data, 'AnticipatoryGSRAUC', 'testing', 2)

ts_ant_gsr, ts_out_gsr = processGSR(df, draw=True, separate=True)



