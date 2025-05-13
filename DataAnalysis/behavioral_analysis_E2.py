import numpy as np
import pandas as pd
from matplotlib.pyplot import legend
from scipy.stats import ttest_ind, ttest_1samp
import pingouin as pg
import seaborn as sns
import matplotlib.pyplot as plt
from utilities.utility_processGSR import processGSR, area_under_curve


# load processed data
df = pd.read_csv('./Data/E2/Preprocessed/processed_data_experiment_cvxeda_E2.csv')
# df = pd.read_csv('./Data/good_learner_data_experiment_cvxeda.csv')
# df = pd.read_csv('./Data/processed_data_modeled.csv')
df['Condition'] = pd.Categorical(df['Condition'], categories=['NoHighStake', 'HighStake'], ordered=True)

# split the data into training and test data
training_data = df[df['Phase'] != 'Test']
test_data = df[df['Phase'] == 'Test']
CA_data = test_data[test_data['TrialType'] == 2]

# manipulation check
print(training_data.groupby(['Condition', 'KeyResponse'])['Reward'].mean())
print(training_data.groupby(['Condition', 'KeyResponse'])['Reward'].std())

# one sample t-tests
reward_ratio = 0.75 / (0.75 + 0.65)
reward_ratio_hs = 1.29 / (1.29 + 1.18)
print(f'[NoHighStake] Random Chance: {ttest_1samp(CA_data[CA_data['Condition'] == 'NoHighStake'].groupby('Subnum')['BestOption'].mean(), 0.50)}')
print(f'[NoHighStake] Reward Ratio: {ttest_1samp(CA_data[CA_data['Condition'] == 'NoHighStake'].groupby('Subnum')['BestOption'].mean(), reward_ratio)}')
print(f'[HighStake] Random Chance: {ttest_1samp(CA_data[CA_data['Condition'] == 'HighStake'].groupby('Subnum')['BestOption'].mean(), 0.50)}')
print(f'[HighStake] Reward Ratio: {ttest_1samp(CA_data[CA_data['Condition'] == 'HighStake'].groupby('Subnum')['BestOption'].mean(), reward_ratio)}')
print(CA_data[CA_data['Condition'] == 'NoHighStake'].groupby('Subnum')['BestOption'].mean().mean())

# calculate the percentage of selecting the best option
prop_optimal = df.groupby(['Subnum', 'Condition', 'TrialType', 'HighStakes'])['BestOption'].mean().reset_index()

# calculate average payoff per trial type
trial_reward = df.groupby(['Condition', 'KeyResponse'],
                          observed=False)['Reward'].mean().reset_index().dropna()

# plot the percentage of selecting the best option
x_labels = ['AB', 'CD', 'CA', 'CB', 'AD', 'BD']

# set the order of the conditions
prop_optimal['Condition'] = pd.Categorical(prop_optimal['Condition'], categories=['NoHighStake', 'HighStake'], ordered=True)

plt.figure(figsize=(10, 8))
sns.barplot(data=prop_optimal, x='TrialType', y='BestOption', hue='Condition', errorbar='se',
            palette=sns.color_palette('deep')[0:2])
plt.xticks(np.arange(6), x_labels, fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('Trial Type', fontsize=20)
plt.ylabel('Proportion of Selecting the Best Option', fontsize=20)
plt.legend(title='Trial Stakes', loc='upper left', fontsize=15, title_fontsize=15)
plt.axhline(0.5, color='black', linestyle='--')
sns.despine()
plt.tight_layout()
plt.savefig('./figures/pre_behavioral_E2.png', dpi=600)
plt.show()

# High Stake only
prop_optimal_hs = prop_optimal[prop_optimal['Condition'] == 'HighStake']
prop_optimal_hs['HighStakes'] = pd.Categorical(prop_optimal_hs['HighStakes'], categories=[False, True],
                                       ordered=True).rename_categories(
    {False: 'Low Stake Trials', True: 'High Stake Trials'})
plt.figure(figsize=(10, 8))
sns.barplot(data=prop_optimal_hs, x='TrialType', y='BestOption', hue='HighStakes', errorbar='se',
            palette=sns.color_palette('deep')[0:2])
plt.xticks(np.arange(6), x_labels, fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('Trial Type', fontsize=20)
plt.ylabel('Proportion of Selecting the Best Option', fontsize=20)
plt.legend(title='Trial Stakes', loc='upper left', fontsize=15, title_fontsize=15)
plt.axhline(0.5, color='black', linestyle='--')
sns.despine()
plt.tight_layout()
plt.savefig('./figures/pre_behavioral_HS_E2.png', dpi=600)
plt.show()

# CA only
CA_summary = CA_data.groupby(['Subnum', 'Condition'])['BestOption'].mean().reset_index()
CA_summary['Condition'] = pd.Categorical(CA_summary['Condition'], categories=['Baseline', 'Frequency', 'Magnitude'], ordered=True)
plt.figure(figsize=(10, 8))
sns.barplot(data=CA_summary, x='Condition', y='BestOption', hue='Condition', errorbar='se',
            palette=sns.color_palette('deep')[0:3])
plt.xticks(fontsize=25)
plt.yticks(fontsize=20)
plt.xlabel('')
plt.ylabel('Proportion of Selecting C', fontsize=25)
plt.ylim(0, 0.7)
plt.axhline(0.5, color='black', linestyle='--')
sns.despine()
plt.tight_layout()
plt.savefig('./figures/pre_behavioral_CA.png', dpi=600)
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

# ======================================================================================================================
# Participant level analysis
# ======================================================================================================================
CA_prob_choice = CA_data.groupby(['Subnum', 'Condition', 'TrialType'])['prob_choice'].mean().reset_index()
CA_prob_choice['Condition'] = pd.Categorical(CA_prob_choice['Condition'], categories=['Baseline', 'Frequency', 'Magnitude'], ordered=True)
sns.barplot(CA_prob_choice, x='Condition', y='prob_choice', hue='Condition', errorbar='se', palette=sns.color_palette('deep')[0:3])
plt.ylabel('Model-Inferred Probability of Choosing C')
plt.xlabel('')
sns.despine()
plt.tight_layout()
plt.savefig('./figures/pre_prob_choice.png', dpi=300)
plt.show()

# make a big plot to show the relationship between the model-inferred probability of choosing C and the anticipatory SCR

sns.lmplot(
    data=df,
    x="dist",
    y="PhasicAnticipatoryGSRAUC",
    col="Condition",      # Create a separate subplot for each condition
    hue="BestOption",    # Color the regression lines/points by correctness
    markers=["o", "s"],   # You can choose different markers for each group
    palette="Set1",       # Choose a palette that suits your needs
    aspect=1.2,           # Adjust the aspect ratio of each facet
    ci=95                 # Confidence interval for the regression lines
)

plt.tight_layout()
plt.savefig('./figures/pre_model_scr.png', dpi=600)
plt.show()