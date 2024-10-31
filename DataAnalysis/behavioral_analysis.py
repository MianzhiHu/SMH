import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
import seaborn as sns
import matplotlib.pyplot as plt
from utilities.utility_processGSR import processGSR, area_under_curve


# load processed data
df = pd.read_csv('./Data/processed_data_cda.csv')

# split the data into training and test data
training_data = df[df['Phase'] != 'Test']
test_data = df[df['Phase'] == 'Test']

# calculate the percentage of selecting the best option
prop_optimal = df.groupby(['Subnum', 'Condition', 'SetSeen '])['BestOption'].mean().reset_index()

# plot the percentage of selecting the best option
x_labels = ['AB', 'CD', 'CA', 'CB', 'AD', 'BD']

# set the order of the conditions
prop_optimal['Condition'] = pd.Categorical(prop_optimal['Condition'], categories=['Baseline', 'Frequency', 'Magnitude'], ordered=True)

plt.figure(figsize=(10, 8))
sns.barplot(data=prop_optimal, x='SetSeen ', y='BestOption', hue='Condition')
plt.xticks(np.arange(6), x_labels)
plt.xlabel('Trial Type')
plt.ylabel('Proportion of Selecting the Best Option')
plt.legend(title='Condition', loc='upper left')
plt.axhline(0.5, color='black', linestyle='--')
sns.despine()
plt.savefig('./figures/pre_behaviroal.png', dpi=300)
plt.show()

