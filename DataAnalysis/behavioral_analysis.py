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

# calculate the percentage of selecting the best option
prop_optimal = df.groupby(['Subnum', 'Condition', 'SetSeen '])['BestOption'].mean().reset_index()

# plot the percentage of selecting the best option
x_labels = ['AB', 'CD', 'CA', 'CB', 'AD', 'BD']

sns.barplot(data=prop_optimal, x='SetSeen ', y='BestOption', hue='Condition')
plt.xticks(np.arange(6), x_labels)
plt.title('Percentage of selecting the best option')
plt.show()


