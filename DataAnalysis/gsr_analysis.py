import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
import seaborn as sns
import matplotlib.pyplot as plt
from utilities.utility_processGSR import processGSR, area_under_curve, pairwise_t_test_GSR
from utilities.utility_plotting import plot_trial, plot_overall, plot_by_phase
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.graphics.factorplots import interaction_plot

# load processed data
path = './Data/processed_data_cda.csv'
df = pd.read_csv(path)

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
pairwise_t_test_GSR(test_data, 'AnticipatoryGSRAUC', 'testing', 2)
pairwise_t_test_GSR(test_data, 'PhasicAnticipatoryGSRAUC', 'testing', 2)
pairwise_t_test_GSR(test_data, 'TonicAnticipatoryGSRAUC', 'testing', 2)

# calculate the variance of the original Iowa Gambling Task
deckA = [100, 100, -50, 100, -200, 100, -100, 100, -150, -250]
deckB = [100, 100, 100, 100, 100, 100, 100, 100, 100, -1150]
deckC = [50, 50, 0, 50, 0, 50, 0, 0, 0, 50]
deckD = [50, 50, 50, 50, 50, 50, 50, 50, 50, -200]
print(f'Deck A: {np.std(deckA)}')
print(f'Deck B: {np.std(deckB)}')
print(f'Deck C: {np.std(deckC)}')
print(f'Deck D: {np.std(deckD)}')

# ======================================================================================================================
#                                                  Advanced Analysis
# ======================================================================================================================
# turn best option into a categorical variable
df['BestOption'] = df['BestOption'].astype('category')
test_CA.loc[:, 'BestOption'] = test_CA['BestOption'].astype('category')
test_CA.loc[:, 'Condition'] = test_CA['Condition'].astype('str')

model = smf.mixedlm("PhasicAnticipatoryGSRAUC ~ BestOption * Condition", test_CA, groups=test_CA["Subnum"]).fit()
print(model.summary())

model = smf.mixedlm("PhasicOutcomeGSRAUC ~ BestOption + Condition", df, groups=df["Subnum"]).fit()
print(model.summary())

# two-way ANOVA
model = smf.ols("PhasicAnticipatoryGSRAUC ~ C(BestOption) + C(Condition) + C(BestOption) * C(Condition)",
                test_CA).fit()
print(sm.stats.anova_lm(model))

sns.pointplot(
    data=test_CA,
    x='BestOption',
    y='PhasicAnticipatoryGSRAUC',
    hue='Condition',
    dodge=True,
    errorbar='se'
)
plt.title('Interaction Plot')
plt.savefig('./figures/interaction_plot.png', dpi=600)
plt.show()


# ======================================================================================================================
#                                                  Plots
# ======================================================================================================================
# plot out the anticipatory GSR data by condition
test_AD = test_data[test_data['SetSeen '] == 4]

for signal in ['PhasicOutcomeGSRAUC', 'PhasicAnticipatoryGSRAUC']:
    plot_trial(test_CA, signal, f'{signal} (uS/sec)', trial="CA")

# plot out the overall anticipatory and outcome GSR data by best option
plot_overall(test_data, 'AUC (uS/sec)', 'train')

# plot out the anticipatory and outcome GSR data by phase
plot_by_phase(df, 'PhasicAnticipatoryGSRAUC', 'Anticipatory AUC (uS/sec)')
