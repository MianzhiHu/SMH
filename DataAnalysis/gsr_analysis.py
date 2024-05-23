import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
import seaborn as sns
import matplotlib.pyplot as plt
from utilities.utility_processGSR import processGSR, area_under_curve, pairwise_t_test_GSR
import statsmodels.formula.api as smf

# load processed data
df = pd.read_csv('./Data/preliminary_data.csv')

# split the data into training and test data
training_data = df[df['Phase'] != 'Test']
test_data = df[df['Phase'] == 'Test']
test_CA = test_data[test_data['SetSeen '] == 2]

# remove trial-level data and save the individual-level data as a separate file for future use
df_individual = df.drop_duplicates(subset='Subnum').reset_index(drop=True)
df_individual = df_individual.drop(columns=['Trial_Index', 'ReactTime', 'Reward', 'BestOption', 'KeyResponse',
                                            'SetSeen ', 'OptionRwdMean', 'Phase', 'AnticipatoryGSRAUC',
                                            'OutcomeGSRAUC', 'PhasicAnticipatoryGSRAUC', 'TonicAnticipatoryGSRAUC',
                                            'PhasicOutcomeGSRAUC', 'TonicOutcomeGSRAUC'])
# df_individual.to_csv('./Data/individual_data.csv', index=False)


# decide if the reward should be perceived as a loss
def calculate_cumulative_average(x):
    return x.expanding().mean().shift(1)  # shift(1) to exclude the current trial's reward


# Apply the function to each participant's data
training_data.loc[:, 'Cumulative_Average'] = training_data.groupby('Subnum')['Reward'].transform(
    calculate_cumulative_average)
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
ttest_ind(training_data[training_data['Loss'] == 0]['OutcomeGSRAUC'],
          training_data[training_data['Loss'] == 1]['OutcomeGSRAUC'])
print('Outcome GSR for the losses:', training_data[training_data['Loss'] == 1]['OutcomeGSRAUC'].mean())
print('Outcome GSR for the wins:', training_data[training_data['Loss'] == 0]['OutcomeGSRAUC'].mean())

# now analyze anticipatory and outcome GSR
# there doesn't seem to be a significant difference
pairwise_t_test_GSR(training_data, 'AnticipatoryGSRAUC', 'training')
pairwise_t_test_GSR(training_data, 'OutcomeGSRAUC', 'training')

# do a simple t-test between BestOption 1 and 0
# there doesn't seem to be a significant difference
ttest_ind(training_data[training_data['BestOption'] == 1]['AnticipatoryGSRAUC'],
          training_data[training_data['BestOption'] == 0]['AnticipatoryGSRAUC'])
print('Anticipatory GSR for the best option:',
      training_data[training_data['BestOption'] == 1]['AnticipatoryGSRAUC'].mean())
print('Anticipatory GSR for the worst option:',
      training_data[training_data['BestOption'] == 0]['AnticipatoryGSRAUC'].mean())

ttest_ind(training_data[training_data['BestOption'] == 1]['OutcomeGSRAUC'],
          training_data[training_data['BestOption'] == 0]['OutcomeGSRAUC'])
print('Outcome GSR for the best option:', training_data[training_data['BestOption'] == 1]['OutcomeGSRAUC'].mean())
print('Outcome GSR for the worst option:', training_data[training_data['BestOption'] == 0]['OutcomeGSRAUC'].mean())

# test data
ttest_ind(test_data[test_data['BestOption'] == 1]['AnticipatoryGSRAUC'],
          test_data[test_data['BestOption'] == 0]['AnticipatoryGSRAUC'])
print('Outcome GSR for the best option:', test_data[test_data['BestOption'] == 1]['AnticipatoryGSRAUC'].mean())
print('Outcome GSR for the worst option:', test_data[test_data['BestOption'] == 0]['AnticipatoryGSRAUC'].mean())

ttest_ind(df[df['BestOption'] == 1]['AnticipatoryGSRAUC'],
          df[df['BestOption'] == 0]['AnticipatoryGSRAUC'])
print('Outcome GSR for the best option:', df[df['BestOption'] == 1]['AnticipatoryGSRAUC'].mean())
print('Outcome GSR for the worst option:', df[df['BestOption'] == 0]['AnticipatoryGSRAUC'].mean())


# CA trials
pairwise_t_test_GSR(test_data, 'AnticipatoryGSRAUC', 'testing', 2)
pairwise_t_test_GSR(test_data, 'PhasicAnticipatoryGSRAUC', 'testing', 2)
pairwise_t_test_GSR(test_data, 'TonicAnticipatoryGSRAUC', 'testing', 2)

# ======================================================================================================================
#                                                  Advanced Analysis
# ======================================================================================================================
# turn best option into a categorical variable
df['BestOption'] = df['BestOption'].astype('category')
test_CA.loc[:, 'BestOption'] = test_CA['BestOption'].astype('category')
test_CA.loc[:, 'Condition'] = test_CA['Condition'].astype('category')

model = smf.mixedlm("TonicAnticipatoryGSRAUC ~ BestOption * Condition", test_CA, groups=test_CA["Subnum"]).fit()
print(model.summary())

model = smf.mixedlm("OutcomeGSRAUC ~ BestOption + Condition", df, groups=df["Subnum"]).fit()
print(model.summary())


# ======================================================================================================================
#                                                  Plots
# ======================================================================================================================
# # plot out the anticipatory GSR data by condition
# plt.figure(figsize=(8, 6))
# sns.set_style("white")
# sns.barplot(data=test_CA, x='Condition', y='PhasicAnticipatoryGSRAUC', hue='BestOption')
# handles, labels = plt.gca().get_legend_handles_labels()
# # customizing the legend
# plt.legend(title='Selected Option', loc='upper left', labels=['A', 'C'], handles=handles)
# plt.xlabel('')
# plt.ylabel('Anticipatory AUC (uS/sec)')
# sns.despine()
# plt.savefig('./figures/pre_PhasicAnticipatoryGSR_CA.png', dpi=300)
# plt.show()
#
#
# plot out the overall anticipatory and outcome GSR data by best option
df_melted = test_data.melt(id_vars='BestOption', value_vars=['AnticipatoryGSRAUC', 'OutcomeGSRAUC'],
                            var_name='GSR_Type', value_name='AUC')

plt.figure(figsize=(8, 6))
sns.set_style("white")
sns.barplot(data=df_melted, x='BestOption', y='AUC', hue='GSR_Type')
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(title='GSR Type', labels=['Anticipatory', 'Outcome'], handles=handles)
plt.xlabel('')
plt.ylabel('AUC (uS/sec)')
sns.despine()
plt.xticks(np.arange(2), ['Suboptimal Option', 'Optimal Option'])
plt.savefig('./figures/test_overall.png', dpi=300)
plt.show()

# # plot out the anticipatory and outcome GSR data by phase
# plt.figure(figsize=(8, 6))
# sns.set_style("white")
# sns.lineplot(data=df, x='Phase', y='AnticipatoryGSRAUC', hue='BestOption')
# handles, labels = plt.gca().get_legend_handles_labels()
# plt.legend(title='Selected Option', loc='upper left', labels=['Suboptimal Option', 'Optimal Option'], handles=handles)
# plt.xlabel('')
# plt.ylabel('Anticipatory AUC (uS/sec)')
# sns.despine()
# plt.savefig('./figures/pre_AntbyPhase.png', dpi=300)
# plt.show()
