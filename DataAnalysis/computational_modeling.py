import ast
import os
import numpy as np
import pandas as pd
from sympy.benchmarks.bench_meijerint import alpha

from utils.ComputationalModeling import ComputationalModels, dict_generator, parameter_extractor, trial_exploder
from utils.DualProcess import DualProcessModel
from scipy.stats import pearsonr, spearmanr, ttest_ind
import statsmodels.formula.api as smf
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

# load processed data
data = pd.read_csv('./Data/processed_data_experiment_cvxeda.csv')

# process the data
data = data.reset_index(drop=True)
data['KeyResponse'] = data['KeyResponse'] - 1
data_copy = data.copy()
data_copy.rename(columns={'TrialType': 'SetSeen.'}, inplace=True)

# generate the dictionary
data_dict = dict_generator(data_copy)

# initialize the computational models
model_dual = DualProcessModel()
model_delta = ComputationalModels('delta')
model_decay = ComputationalModels('decay')

if __name__ == '__main__':
    # # fit the models
    # dual_results = model_dual.fit(data_dict, 'Entropy_Dis_ID', Gau_fun='Naive_Recency', Dir_fun='Linear_Recency',
    #                               weight_Dir='softmax', weight_Gau='softmax', num_iterations=100, a_min=1e-32)
    # dual_results.to_csv('./models/dual_results.csv', index=False)
    #
    # delta_results = model_delta.fit(data_dict, num_iterations=100)
    # delta_results.to_csv('./models/delta_results.csv', index=False)
    #
    # decay_results = model_decay.fit(data_dict, num_iterations=100)
    # decay_results.to_csv('./models/decay_results.csv', index=False)

    dual_results = pd.read_csv('./models/dual_results.csv')
    delta_results = pd.read_csv('./models/delta_results.csv')
    decay_results = pd.read_csv('./models/decay_results.csv')

    # compare the models
    print(f'[AIC] Dual: {dual_results["AIC"].mean()} Delta: {delta_results["AIC"].mean()} '
          f'Decay: {decay_results["AIC"].mean()}')
    print(f'[BIC] Dual: {dual_results["BIC"].mean()} Delta: {delta_results["BIC"].mean()} '
          f'Decay: {decay_results["BIC"].mean()}')

    # check how many participants are best fit by each model
    dual_results['delta_AIC'] = delta_results['AIC']
    dual_results['decay_AIC'] = decay_results['AIC']
    dual_results['delta_BIC'] = delta_results['BIC']
    dual_results['decay_BIC'] = decay_results['BIC']
    dual_results['best_model'] = dual_results[['BIC', 'delta_BIC', 'decay_BIC']].idxmin(axis=1)
    print(dual_results['best_model'].value_counts())

    # extract the parameters
    dual_results = parameter_extractor(dual_results)

    # extract the weights
    data['best_weight'] = trial_exploder(dual_results, 'best_weight')
    data['best_obj_weight'] = trial_exploder(dual_results, 'best_obj_weight')
    data['best_gau_entropy'] = trial_exploder(dual_results, 'best_gau_entropy')
    data['best_dir_entropy'] = trial_exploder(dual_results, 'best_dir_entropy')
    data['prob_best'] = trial_exploder(dual_results, 'best_prob_choice')

    # if BestOption is 1, then set "Prob Best' to prob choice, else set it to 1 - prob choice
    data['prob_choice'] = np.where(data['BestOption'] == 1, data['prob_best'], 1 - data['prob_best'])
    data['pred_choice'] = np.where(data['prob_best'] > 0.5, 1, 0)
    data['mismatch'] = np.where(data['pred_choice'] != data['BestOption'], 1, 0)

    # clean up the columns
    col_to_keep = ['participant_id', 'AIC', 'BIC', 't', 'alpha', 'subj_weight']
    dual_results = dual_results[col_to_keep]
    dual_results.rename(columns={'participant_id': 'Subnum'}, inplace=True)

    # merge the data
    data = data.merge(dual_results, on='Subnum')
    data['dist'] = abs(data['prob_choice'] - 0.5)

    # save the data
    data.to_csv('./Data/processed_data_modeled.csv', index=False)

    # Now group by trial type and choice to see the difference in anticipatory GSR
    PhasicAnticipatoryGSRAUC_grouped = data.groupby(['Subnum', 'TrialType', 'Condition',  'BestOption'])['PhasicAnticipatoryGSRAUC'].mean().reset_index()
    PhasicAnticipatoryGSRAUC_grouped = PhasicAnticipatoryGSRAUC_grouped.pivot(index=['Subnum', 'TrialType', 'Condition'], columns='BestOption',
                                      values='PhasicAnticipatoryGSRAUC').reset_index()
    PhasicAnticipatoryGSRAUC_grouped.columns = ['Subnum', 'TrialType', 'Condition', 'Optimal', 'Suboptimal']
    CA_AAUC = PhasicAnticipatoryGSRAUC_grouped[PhasicAnticipatoryGSRAUC_grouped['TrialType'] == 2]
    CA_AAUC.loc[:, 'GSRdiff'] = CA_AAUC['Suboptimal'] - CA_AAUC['Optimal']

    best_weight_grouped = data.groupby(['Subnum', 'TrialType', 'Condition'])['best_weight'].mean().reset_index()
    best_weight_grouped = best_weight_grouped[best_weight_grouped['TrialType'] == 2]

    best_prop = data.groupby(['Subnum', 'TrialType', 'Condition'])['BestOption'].mean().reset_index()
    best_prop = best_prop[best_prop['TrialType'] == 2]


    results = dual_results.merge(CA_AAUC[['Subnum', 'Condition', 'GSRdiff']], on='Subnum')
    results = results.merge(best_weight_grouped [['Subnum', 'best_weight']], on='Subnum')
    results = results.merge(best_prop[['Subnum', 'BestOption']], on='Subnum')
    results = results.dropna()
    results['best_weight'] = pd.to_numeric(results['best_weight'], errors='coerce')

    numeric_col = ['PhasicAnticipatoryGSRAUC', 'best_gau_entropy', 'best_dir_entropy', 'best_weight', 'best_obj_weight',
                   'dist', 'subj_weight', 'alpha', 't']

    for col in numeric_col:
        data[col] = pd.to_numeric(data[col], errors='coerce')


    # pearson r
    print(pearsonr(results['GSRdiff'], results['best_weight']))
    print(pearsonr(data['PhasicAnticipatoryGSRAUC'], data['best_gau_entropy']))

    # separate by condition
    baseline = results[results['Condition'] == 'Baseline']
    frequency = results[results['Condition'] == 'Frequency']
    magnitude = results[results['Condition'] == 'Magnitude']
    baseline_data = data[data['Condition'] == 'Baseline']
    frequency_data = data[data['Condition'] == 'Frequency']
    magnitude_data = data[data['Condition'] == 'Magnitude']

    # t-test
    print(pearsonr(baseline['GSRdiff'], baseline['best_weight']))
    print(pearsonr(frequency['GSRdiff'], frequency['best_weight']))
    print(pearsonr(magnitude['GSRdiff'], magnitude['best_weight']))
    print(pearsonr(baseline_data['PhasicAnticipatoryGSRAUC'], baseline_data['dist']))
    print(pearsonr(frequency_data['PhasicAnticipatoryGSRAUC'], frequency_data['dist']))
    print(pearsonr(magnitude_data['PhasicAnticipatoryGSRAUC'], magnitude_data['dist']))

    # mixed effects model
    mixed_model = smf.mixedlm('PhasicAnticipatoryGSRAUC ~ best_gau_entropy + C(Condition)', data=data, groups=data['Subnum']).fit()
    print(mixed_model.summary())

    # ==================================================================================================================
    # Simulate the model
    # ==================================================================================================================
    val = [1.95, 1.05, 2.25, 0.75]
    var_mag = [1.29, 2.58, 2.58, 1.29]
    var_baseline = [1.29, 1.29, 1.29, 1.29]
    n_iterations = 10000

    # # simulate the data
    # dual_baseline = model_dual.simulate(val, var_baseline, model="Entropy_Dis_ID", AB_freq=75, CD_freq=75,
    #                                     num_iterations=n_iterations, weight_Gau='softmax', weight_Dir='softmax',
    #                                     arbi_option='Entropy', Dir_fun='Linear_Recency', Gau_fun='Naive_Recency', a_min=1e-32)
    # dual_frequency = model_dual.simulate(val, var_baseline, model="Entropy_Dis_ID", AB_freq=100, CD_freq=50,
    #                                      num_iterations=n_iterations, weight_Gau='softmax', weight_Dir='softmax',
    #                                      arbi_option='Entropy', Dir_fun='Linear_Recency', Gau_fun='Naive_Recency', a_min=1e-32)
    # dual_magnitude = model_dual.simulate(val, var_mag, model="Entropy_Dis_ID", AB_freq=75, CD_freq=75,
    #                                      num_iterations=n_iterations, weight_Gau='softmax', weight_Dir='softmax',
    #                                      arbi_option='Entropy', Dir_fun='Linear_Recency', Gau_fun='Naive_Recency', a_min=1e-32)
    #
    # dual_baseline.to_csv('./Data/Model/Simulations/dual_baseline.csv', index=False)
    # dual_frequency.to_csv('./Data/Model/Simulations/dual_frequency.csv', index=False)
    # dual_magnitude.to_csv('./Data/Model/Simulations/dual_magnitude.csv', index=False)

    # # simulate for the delta model
    # delta_baseline = model_delta.simulate(val, var_baseline, AB_freq=75, CD_freq=75, num_iterations=n_iterations)
    # delta_frequency = model_delta.simulate(val, var_baseline, AB_freq=100, CD_freq=50, num_iterations=n_iterations)
    # delta_magnitude = model_delta.simulate(val, var_mag, AB_freq=75, CD_freq=75, num_iterations=n_iterations)
    #
    # delta_baseline.to_csv('./Data/Model/Simulations/delta_baseline.csv', index=False)
    # delta_frequency.to_csv('./Data/Model/Simulations/delta_frequency.csv', index=False)
    # delta_magnitude.to_csv('./Data/Model/Simulations/delta_magnitude.csv', index=False)
    #
    # # simulate for the decay model
    # decay_baseline = model_decay.simulate(val, var_baseline, AB_freq=75, CD_freq=75, num_iterations=n_iterations)
    # decay_frequency = model_decay.simulate(val, var_baseline, AB_freq=100, CD_freq=50, num_iterations=n_iterations)
    # decay_magnitude = model_decay.simulate(val, var_mag, AB_freq=75, CD_freq=75, num_iterations=n_iterations)
    #
    # decay_baseline.to_csv('./Data/Model/Simulations/decay_baseline.csv', index=False)
    # decay_frequency.to_csv('./Data/Model/Simulations/decay_frequency.csv', index=False)
    # decay_magnitude.to_csv('./Data/Model/Simulations/decay_magnitude.csv', index=False)

    # ------------------------------------------------------------------------------------------------------------------
    # Read the simulated data
    # ------------------------------------------------------------------------------------------------------------------
    # Load the data
    folder_path = './Data/Model/Simulations/'
    best_option_mappping = {
        'AB': 'A',
        'CD': 'C',
        'CA': 'C',
        'BD': 'B',
        'AD': 'A',
        'CB': 'C'
    }
    all_sim = {}

    for file in os.listdir(folder_path):
        if file.endswith('.csv'):
            file_path = os.path.join(folder_path, file)
            all_sim[file] = pd.read_csv(file_path)
            all_sim[file].loc[:, 'Model'] = os.path.splitext(file)[0].split('_')[0]
            all_sim[file].loc[:, 'Condition'] = os.path.splitext(file)[0].split('_')[1]
            all_sim[file].loc[:, 'pair'] = all_sim[file]['pair'].apply(ast.literal_eval)
            all_sim[file].loc[:, 'pair'] = all_sim[file]['pair'].apply(lambda x: ''.join(x))
            all_sim[file].loc[:, 'mapping'] = all_sim[file]['pair'].apply(lambda x: best_option_mappping[x])
            all_sim[file].loc[:, 'bestoption'] = (
                        all_sim[file]['choice'] == all_sim[file]['mapping']).astype(int)

    # combine the data
    all_sim_df = pd.concat(all_sim.values())
    print(all_sim_df['Model'].unique())
    # all_sim_df['Model'] = pd.Categorical(all_sim_df['Model'], categories=['delta', 'decay', 'dual'], ordered=True)

    # plot the data
    CA = all_sim_df[all_sim_df['pair'] == 'CA']
    plt.figure(figsize=(10, 8))
    sns.barplot(data=CA, x='Model', y='bestoption', hue='Condition', errorbar=None, palette=sns.color_palette('deep')[0:3])
    plt.ylabel('Proportion of Selecting the Best Option', fontsize=25)
    plt.xlabel('')
    plt.xticks(labels=['Delta', 'Dual-Process'], ticks=[0, 1], fontsize=25)
    plt.yticks(fontsize=20)
    plt.ylim(0, 0.7)
    handles, labels = plt.gca().get_legend_handles_labels()
    label_map = {
        "baseline": "Baseline",
        "frequency": "Frequency",
        "magnitude": "Variance"
    }
    new_labels = [label_map.get(lbl, lbl) for lbl in labels]
    plt.legend(handles, new_labels, title='Condition', fontsize=20, title_fontsize=20, loc='lower left')
    sns.despine()
    plt.tight_layout()
    plt.savefig('./figures/simulated_CA.png', dpi=600)
    plt.show()


