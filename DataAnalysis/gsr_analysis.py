import numpy as np
import pandas as pd
from brainsmash.mapgen.stats import pearsonr
from scipy.stats import ttest_ind
import seaborn as sns
import matplotlib.pyplot as plt
from sympy.physics.units import frequency
from utilities.utility_processGSR import processGSR, area_under_curve, pairwise_t_test_GSR
from utilities.utility_plotting import plot_trial, plot_overall, plot_by_phase
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy.stats import pearsonr
import pingouin as pg

# load processed data
# path = './Data/processed_data_modeled.csv'

# path = './Data/processed_data_auto.csv'
# path = './Data/processed_data_trial_auto.csv'
# path = './Data/processed_data_difference.csv'
# path = './Data/processed_data_trial_highpass.csv'
# path = './Data/processed_data_experiment_smoothmedian.csv'
# path = './Data/processed_data_trial_smoothmedian.csv'
path = './Data/processed_data_experiment_cvxeda.csv'
# path = './Data/processed_data_trial_cvxeda.csv'
# path = './Data/good_learner_data_experiment_cvxeda.csv'
# path = './Data/processed_data_experiment_cda.csv'
# path = './Data/processed_data_trial_cda.csv'
# path = './Data/processed_data_combined.csv'
# path = './Data/processed_data_experiment_SparsEDA.csv'
# path = './Data/processed_data_trial_SparsEDA.csv'

df = pd.read_csv(path)

# # taking z-score potentially
# all_auc_columns = ['AnticipatoryGSRAUC', 'OutcomeGSRAUC', 'TonicAnticipatoryGSRAUC', 'PhasicAnticipatoryGSRAUC',
#                    'TonicOutcomeGSRAUC', 'PhasicOutcomeGSRAUC', 'GSRAUC', 'TonicGSRAUC', 'PhasicGSRAUC']
# for auc_column in all_auc_columns:
#     df[auc_column] = df.groupby('Subnum')[auc_column].transform(lambda x: (x - x.mean()) / x.std())

# remove nan values
df = df.dropna()
df['Condition'] = pd.Categorical(df['Condition'], categories=['Baseline', 'Frequency', 'Magnitude'], ordered=True)

# split the data into training and test data
training_data = df[df['Phase'] != 'Test']
test_data = df[df['Phase'] == 'Test']
train_AB = training_data[training_data['TrialType'] == 0]
train_CD = training_data[training_data['TrialType'] == 1]
test_CA = test_data[test_data['TrialType'] == 2]
test_CB = test_data[test_data['TrialType'] == 3]
test_AD = test_data[test_data['TrialType'] == 4]
test_BD = test_data[test_data['TrialType'] == 5]

# remove trial-level data and save the individual-level data as a separate file for future use
df_individual = df.drop_duplicates(subset='Subnum').reset_index(drop=True)
df_individual = df_individual.drop(columns=['Trial_Index', 'ReactTime', 'Reward', 'BestOption', 'KeyResponse',
                                            'TrialType', 'OptionRwdMean', 'Phase', 'AnticipatoryGSRAUC',
                                            'OutcomeGSRAUC', 'PhasicAnticipatoryGSRAUC', 'TonicAnticipatoryGSRAUC',
                                            'PhasicOutcomeGSRAUC', 'TonicOutcomeGSRAUC'])
df_individual.to_csv('./Data/individual_data.csv', index=False)


# decide if the reward should be perceived as a loss
def calculate_cumulative_average(x):
    return x.expanding().mean().shift(1)  # shift(1) to exclude the current trial's reward


# Apply the function to each participant's data
training_data.loc[:, 'Cumulative_Average'] = training_data.groupby('Subnum')['Reward'].transform(
    calculate_cumulative_average)
training_data.loc[:, 'Loss'] = (training_data['Cumulative_Average'] > training_data['Reward']).astype(int)

# record win stay lose shift
# track the previous trial's reward by trial type
training_data.loc[:, 'Previous_Loss'] = training_data.groupby(['Subnum', 'TrialType'])['Loss'].shift(1).copy()
training_data.loc[:, 'Previous_Choice'] = training_data.groupby(['Subnum', 'TrialType'])['KeyResponse'].shift(1).copy()
training_data.loc[:, 'WSLS'] = ((training_data['Previous_Loss'] == 0) & (training_data['KeyResponse'] == training_data['Previous_Choice']) |
                                (training_data['Previous_Loss'] == 1) & (training_data['KeyResponse'] != training_data['Previous_Choice'])).astype(int).copy()
print(training_data.groupby('Condition')['WSLS'].mean())

# further split the data by condition
baseline = df[df['Condition'] == 'Baseline']
baseline_training = training_data[training_data['Condition'] == 'Baseline']
baseline_testing = test_data[test_data['Condition'] == 'Baseline']
print(baseline_training['Subnum'].nunique())

frequency = df[df['Condition'] == 'Frequency']
frequency_training = training_data[training_data['Condition'] == 'Frequency']
frequency_testing = test_data[test_data['Condition'] == 'Frequency']
print(frequency_training['Subnum'].nunique())

magnitude = df[df['Condition'] == 'Magnitude']
magnitude_training = training_data[training_data['Condition'] == 'Magnitude']
magnitude_testing = test_data[test_data['Condition'] == 'Magnitude']
print(magnitude_training['Subnum'].nunique())

# test whether people show higher outcome GSR when they choose the bad option
# there doesn't seem to be a significant difference
ttest_ind(training_data[training_data['Loss'] == 0]['PhasicOutcomeGSRAUC'],
          training_data[training_data['Loss'] == 1]['PhasicOutcomeGSRAUC'])
print('Outcome GSR for the losses:', training_data[training_data['Loss'] == 1]['PhasicOutcomeGSRAUC'].mean())
print('Outcome GSR for the wins:', training_data[training_data['Loss'] == 0]['PhasicOutcomeGSRAUC'].mean())

# now analyze anticipatory and outcome GSR
# there doesn't seem to be a significant difference
pairwise_t_test_GSR(training_data, 'AnticipatoryGSRAUC', 'training')
pairwise_t_test_GSR(training_data, 'OutcomeGSRAUC', 'training')

# do a simple t-test between BestOption 1 and 0
# there doesn't seem to be a significant difference
ttest_ind(training_data[training_data['BestOption'] == 1]['PhasicAnticipatoryGSRAUC'],
          training_data[training_data['BestOption'] == 0]['PhasicAnticipatoryGSRAUC'])
print('Anticipatory GSR for the best option:',
      training_data[training_data['BestOption'] == 1]['PhasicAnticipatoryGSRAUC'].mean())
print('Anticipatory GSR for the worst option:',
      training_data[training_data['BestOption'] == 0]['PhasicAnticipatoryGSRAUC'].mean())

ttest_ind(training_data[training_data['BestOption'] == 1]['PhasicOutcomeGSRAUC'],
          training_data[training_data['BestOption'] == 0]['PhasicOutcomeGSRAUC'])
print('Outcome GSR for the best option:', training_data[training_data['BestOption'] == 1]['PhasicOutcomeGSRAUC'].mean())
print('Outcome GSR for the worst option:', training_data[training_data['BestOption'] == 0]['PhasicOutcomeGSRAUC'].mean())

# test data
ttest_ind(test_data[test_data['BestOption'] == 1]['PhasicAnticipatoryGSRAUC'],
          test_data[test_data['BestOption'] == 0]['PhasicAnticipatoryGSRAUC'])
print('Outcome GSR for the best option:', test_data[test_data['BestOption'] == 1]['PhasicAnticipatoryGSRAUC'].mean())
print('Outcome GSR for the worst option:', test_data[test_data['BestOption'] == 0]['PhasicAnticipatoryGSRAUC'].mean())

ttest_ind(df[df['BestOption'] == 1]['PhasicAnticipatoryGSRAUC'],
          df[df['BestOption'] == 0]['PhasicAnticipatoryGSRAUC'])
print('Outcome GSR for the best option:', df[df['BestOption'] == 1]['PhasicAnticipatoryGSRAUC'].mean())
print('Outcome GSR for the worst option:', df[df['BestOption'] == 0]['PhasicAnticipatoryGSRAUC'].mean())


# CA trials
pairwise_t_test_GSR(test_data, 'PhasicAnticipatoryGSRAUC', 'testing', 2)

# t-test between conditions
print(ttest_ind(df[df['Condition'] == 'Baseline'].groupby('Subnum')['PhasicAnticipatoryGSRAUC'].mean(),
                df[df['Condition'] == 'Frequency'].groupby('Subnum')['PhasicAnticipatoryGSRAUC'].mean()))
print(f'Baseline Phasic: {df[df["Condition"] == "Baseline"]["PhasicAnticipatoryGSRAUC"].mean()}')
print(f'Frequency Phasic: {df[df["Condition"] == "Frequency"]["PhasicAnticipatoryGSRAUC"].mean()}')
print(f'Magnitude Phasic: {df[df["Condition"] == "Magnitude"]["PhasicAnticipatoryGSRAUC"].mean()}')
print(f'Baseline Tonic: {df[df["Condition"] == "Baseline"]["TonicAnticipatoryGSRAUC"].mean()}')
print(f'Frequency Tonic: {df[df["Condition"] == "Frequency"]["TonicAnticipatoryGSRAUC"].mean()}')
print(f'Magnitude Tonic: {df[df["Condition"] == "Magnitude"]["TonicAnticipatoryGSRAUC"].mean()}')

# ======================================================================================================================
#                                                  Group Analysis
# ======================================================================================================================
# calculate the average C choice rate for each participant
prop_optimal = df.groupby(['Subnum', 'Condition', 'TrialType'])['BestOption'].mean().reset_index()
prop_optimal_CA = prop_optimal[prop_optimal['TrialType'] == 2]
group_mask_A = prop_optimal_CA['BestOption'] < 0.25
group_mask_C = prop_optimal_CA['BestOption'] > 0.75
group_mask_hesitant = (prop_optimal_CA['BestOption'] >= 0.25) & (prop_optimal_CA['BestOption'] <= 0.75)
df['Group'] = df['Subnum'].map(
    {subnum: 'A' if mask_A else 'C' if mask_C else 'H' for subnum, mask_A, mask_C in
        zip(prop_optimal_CA['Subnum'], group_mask_A, group_mask_C)})
df.to_csv('./Data/processed_data_modeled.csv', index=False)

# filter df by the group mask
group_A = df[df['Subnum'].isin(prop_optimal_CA[group_mask_A]['Subnum'].values)]
group_C = df[df['Subnum'].isin(prop_optimal_CA[group_mask_C]['Subnum'].values)]
group_hesitant = df[df['Subnum'].isin(prop_optimal_CA[group_mask_hesitant]['Subnum'].values)]

# Predict the best option using the anticipatory GSR
group = group_A[group_A['TrialType'] == 2 & (group_A['Condition'] == 'Frequency')]
print(group['Condition'].value_counts() / 25)
model = smf.mixedlm("PhasicAnticipatoryGSRAUC ~ C(BestOption)", group, groups=group["Subnum"]).fit()
print(model.summary())


# # ======================================================================================================================
# #                                                  Advanced Analysis
# # ======================================================================================================================
# # turn best option into a categorical variable
condition_of_interest = baseline
df['BestOption'] = df['BestOption'].astype('category')
test_CA.loc[:, 'BestOption'] = test_CA['BestOption'].astype('category')
test_CA.loc[:, 'Condition'] = test_CA['Condition'].astype('str')
#
# model = smf.mixedlm("PhasicAnticipatoryGSRAUC ~ Phase * KeyResponse", baseline_training, groups=baseline_training["Subnum"]).fit()
# print(model.summary())
#
# model = smf.mixedlm("PhasicOutcomeGSRAUC ~ BestOption + Condition", training_data, groups=training_data["Subnum"]).fit()
# print(model.summary())
#
# model = smf.mixedlm("PhasicAnticipatoryGSRAUC ~ best_weight + BestOption", magnitude_testing, groups=magnitude_testing["Subnum"]).fit()
# print(model.summary())
#
# model = smf.mixedlm("best_weight ~ C(Condition)", df, groups=df["Subnum"]).fit()
# print(model.summary())

#
# model = smf.mixedlm("PhasicAnticipatoryGSRAUC ~ C(Condition)", df, groups=df["Subnum"]).fit()
# print(model.summary())
#
# model = smf.mixedlm("PhasicAnticipatoryGSRAUC ~ best_weight * Condition", df, groups=df["Subnum"]).fit()
# print(model.summary())

# model = smf.mixedlm("PhasicAnticipatoryGSRAUC ~ best_weight", condition_of_interest, groups=condition_of_interest["Subnum"]).fit()
# print(model.summary())

# model = smf.mixedlm("best_weight ~ PhasicAnticipatoryGSRAUC + I(PhasicAnticipatoryGSRAUC ** 2) * C(Condition)", df, groups=df["Subnum"]).fit()
# print(model.summary())
#
# #
# # two-way ANOVA
# model = smf.ols("PhasicAnticipatoryGSRAUC ~ C(BestOption) + C(Condition) + C(BestOption) * C(Condition)",
#                 test_CA).fit()
# print(model.summary())
# print(sm.stats.anova_lm(model))

model = pg.mixed_anova(data=test_CA, dv='PhasicAnticipatoryGSRAUC', within='BestOption', between='Condition', subject='Subnum')

# # ======================================================================================================================
# #                                                  Plots
# # ======================================================================================================================
# visualize the overall distribution of the GSR data
fig, ax = plt.subplots(1, 2, figsize=(20, 8))
sns.histplot(df['PhasicAnticipatoryGSRAUC'], bins=20, kde=True, ax=ax[0])
sns.histplot(df['PhasicOutcomeGSRAUC'], bins=20, kde=True, ax=ax[1])
ax[0].set_title('Anticipatory GSR AUC (uS/sec)', fontsize=25)
ax[1].set_title('Outcome GSR AUC (uS/sec)', fontsize=25)
plt.savefig('./figures/GSR_distribution.png', dpi=600)
plt.clf()

# plot the histgram distribution of subjective weights by condition
fig, ax = plt.subplots(1, 3, figsize=(20, 8))
sns.histplot(df[df['Condition'] == 'Baseline']['subj_weight'], bins=20, kde=True, ax=ax[0])
sns.histplot(df[df['Condition'] == 'Frequency']['subj_weight'], bins=20, kde=True, ax=ax[1])
sns.histplot(df[df['Condition'] == 'Magnitude']['subj_weight'], bins=20, kde=True, ax=ax[2])
ax[0].set_title('Baseline', fontsize=25)
ax[1].set_title('Frequency', fontsize=25)
ax[2].set_title('Magnitude', fontsize=25)
plt.savefig('./figures/weight_distribution.png', dpi=600)
plt.clf()

# plot the interaction for each trial type
data_list = [train_AB, train_CD, test_CA, test_CB, test_AD, test_BD]
mapping = {0: 'AB', 1: 'CD', 2: 'CA', 3: 'CB', 4: 'AD', 5: 'BD'}
for trial_type in data_list:
    sns.pointplot(
        data=trial_type,
        x='BestOption',
        y='PhasicAnticipatoryGSRAUC',
        hue='Condition',
        dodge=True,
        errorbar='se'
    )
    plt.title(mapping[trial_type['TrialType'].iloc[0]])
    plt.savefig(f'./figures/Anticipatory_interaction_plot_{mapping[trial_type["TrialType"].iloc[0]]}.png', dpi=600)
    plt.clf()

for trial_type in data_list:
    sns.pointplot(
        data=trial_type,
        x='BestOption',
        y='PhasicOutcomeGSRAUC',
        hue='Condition',
        dodge=True,
        errorbar='se'
    )
    plt.title(mapping[trial_type['TrialType'].iloc[0]])
    plt.savefig(f'./figures/Outcome_interaction_plot_{mapping[trial_type["TrialType"].iloc[0]]}.png', dpi=600)
    plt.clf()

# specifically for the CA trials
df_name = test_CA
df_name = df_name[df_name['TrialType'] == 2]
print(df_name['Condition'].unique())
df_name['Condition'] = pd.Categorical(df_name['Condition'], categories=['Baseline', 'Frequency', 'Magnitude'], ordered=True)
df_name['Condition'] = df_name['Condition'].cat.rename_categories({
    'Magnitude': 'Variance'
})
plt.figure(figsize=(10, 8))
sns.pointplot(
    data=df_name,
    x='BestOption',
    y='PhasicAnticipatoryGSRAUC',
    hue='Condition',
    dodge=True,
    errorbar='se',
    palette=sns.color_palette('deep')[0:3]
)
plt.title('')
plt.ylabel('Anticipatory GSR AUC (uS/sec)', fontsize=25)
plt.xlabel('')
plt.xticks([0, 1], ['A', 'C'], fontsize=25)
plt.yticks(fontsize=20)
plt.legend(title='Condition', fontsize=20, title_fontsize=20, framealpha=0.5)
sns.despine()
plt.savefig(f'./figures/GSR_CA.png', dpi=600)
plt.clf()

# plot out the anticipatory GSR data by condition
for signal in ['PhasicOutcomeGSRAUC', 'PhasicAnticipatoryGSRAUC']:
    plot_trial(test_CA, signal, f'{signal} (uS/sec)', trial="CA")

# plot out the overall anticipatory and outcome GSR data by best option
plot_overall(test_data, 'AUC (uS/sec)', 'test')

# plot out the anticipatory and outcome GSR data by phase
plot_by_phase(frequency_training, 'PhasicAnticipatoryGSRAUC', 'Anticipatory AUC (uS/sec)')
