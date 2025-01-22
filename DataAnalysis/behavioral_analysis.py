import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, ttest_1samp
import seaborn as sns
import matplotlib.pyplot as plt
from utilities.utility_processGSR import processGSR, area_under_curve


# load processed data
df = pd.read_csv('./Data/processed_data_experiment_cvxeda.csv')

# split the data into training and test data
training_data = df[df['Phase'] != 'Test']
test_data = df[df['Phase'] == 'Test']
CA_data = test_data[test_data['TrialType'] == 2]

# manipulation check
print(training_data.groupby(['Condition', 'KeyResponse'])['Reward'].mean())
print(training_data.groupby(['Condition', 'KeyResponse'])['Reward'].std())

# one sample t-tests
print(f'[Baseline] {ttest_1samp(CA_data[CA_data['Condition'] == 'Baseline'].groupby('Subnum')['BestOption'].mean(), 0.5)}')
print(f'[Frequency] {ttest_1samp(CA_data[CA_data['Condition'] == 'Frequency'].groupby('Subnum')['BestOption'].mean(), 0.5)}')
print(f'[Magnitude] {ttest_1samp(CA_data[CA_data['Condition'] == 'Magnitude'].groupby('Subnum')['BestOption'].mean(), 0.5)}')


# calculate the percentage of selecting the best option
prop_optimal = df.groupby(['Subnum', 'Condition', 'TrialType'])['BestOption'].mean().reset_index()

# plot the percentage of selecting the best option
x_labels = ['AB', 'CD', 'CA', 'CB', 'AD', 'BD']

# set the order of the conditions
prop_optimal['Condition'] = pd.Categorical(prop_optimal['Condition'], categories=['Baseline', 'Frequency', 'Magnitude'], ordered=True)

plt.figure(figsize=(10, 8))
sns.barplot(data=prop_optimal, x='TrialType', y='BestOption', hue='Condition', errorbar='se')
plt.xticks(np.arange(6), x_labels)
plt.xlabel('Trial Type')
plt.ylabel('Proportion of Selecting the Best Option')
plt.legend(title='Condition', loc='upper left')
plt.axhline(0.5, color='black', linestyle='--')
sns.despine()
plt.savefig('./figures/pre_behaviroal.png', dpi=300)
plt.show()

# show the distribution of the best option by trial type
trial_type_mapping = {0: 'AB', 1: 'CD', 2: 'CA', 3: 'CB', 4: 'AD', 5: 'BD'}
prop_optimal['Trial Type'] = prop_optimal['TrialType'].map(trial_type_mapping)

sns.set_style("white")
g = sns.FacetGrid(prop_optimal, col='Trial Type', row='Condition', margin_titles=True)
g.map(sns.histplot, 'BestOption', bins=10)
g.set_axis_labels('% of Selecting the Optimal Option', 'Count')
plt.savefig('./figures/pre_best_option_dist.png', dpi=300)
plt.show()