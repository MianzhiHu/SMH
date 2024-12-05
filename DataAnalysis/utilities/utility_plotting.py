import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# Function to plot the anticipatory GSR data in CA trials by condition
def plot_trial(data, signal='PhasicAnticipatoryGSRAUC', ylabel='Anticipatory AUC (uS/sec)', trial="CA"):
    plt.figure(figsize=(8, 6))
    sns.set_style("white")
    sns.barplot(data=data, x='Condition', y=signal, hue='BestOption')
    plt.xlabel('')
    plt.ylabel(ylabel)
    handles, _ = plt.gca().get_legend_handles_labels()
    # get the labels by separating the trial argument
    label_1, label_2 = trial[0], trial[1]
    plt.legend(title='Selected Option', loc='upper left', labels=[label_2, label_1], handles=handles)
    sns.despine()
    plt.savefig(f'./figures/pre_{signal}_{trial}.png', dpi=300)
    plt.show()


# Function to plot the overall anticipatory and outcome GSR data by best option
def plot_overall(data, ylabel='AUC (uS/sec)', phase='test'):
    df_melted = data.melt(id_vars='BestOption', value_vars=['PhasicAnticipatoryGSRAUC', 'PhasicOutcomeGSRAUC'],
                          var_name='GSR_Type', value_name='AUC')

    plt.figure(figsize=(8, 6))
    sns.set_style("white")
    sns.barplot(data=df_melted, x='BestOption', y='AUC', hue='GSR_Type')
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(title='GSR Type', labels=['Anticipatory', 'Outcome'], handles=handles)
    plt.xlabel('')
    plt.ylabel(ylabel)
    sns.despine()
    plt.xticks(np.arange(2), ['Suboptimal Option', 'Optimal Option'])
    plt.savefig(f'./figures/pre_{phase}_overall.png', dpi=300)
    plt.show()


# Function to plot the anticipatory and outcome GSR data by phase
def plot_by_phase(data, signal='PhasicAnticipatoryGSRAUC', ylabel='Anticipatory AUC (uS/sec)'):
    plt.figure(figsize=(8, 6))
    sns.set_style("white")
    sns.lineplot(data=data, x='Phase', y=signal, hue='KeyResponse', errorbar='se')
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(title='Selected Option', loc='upper left', labels=['A', 'B', 'C', 'D'], handles=handles)
    plt.xlabel('')
    plt.ylabel(ylabel)
    sns.despine()
    plt.savefig(f'./figures/pre_{signal}_byPhase.png', dpi=300)
    plt.show()
