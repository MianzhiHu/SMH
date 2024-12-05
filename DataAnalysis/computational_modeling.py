import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.ComputationalModeling import ComputationalModels, dict_generator, parameter_extractor, trial_exploder
from utils.DualProcess import DualProcessModel
from scipy.stats import pearsonr, spearmanr, ttest_ind
import statsmodels.formula.api as smf
import statsmodels.api as sm

# load processed data
data = pd.read_csv('./Data/processed_data_experiment_cda.csv')

# process the data
data = data.reset_index(drop=True)
data['KeyResponse'] = data['KeyResponse'] - 1
data.rename(columns={'SetSeen ': 'SetSeen.'}, inplace=True)

# generate the dictionary
data_dict = dict_generator(data)

# initialize the computational models
model_dual = DualProcessModel()
model_delta = ComputationalModels('delta')
model_decay = ComputationalModels('decay')

if __name__ == '__main__':
    # # fit the models
    # dual_results = model_dual.fit(data_dict, 'Entropy_Dis_ID', Gau_fun='Naive_Recency', Dir_fun='Linear_Recency',
    #                               weight_Dir='softmax', weight_Gau='softmax', num_iterations=200)
    # dual_results.to_csv('./models/dual_results.csv', index=False)

    # delta_results = model_delta.fit(data_dict, num_iterations=200)
    # delta_results.to_csv('./models/delta_results.csv', index=False)
    #
    # decay_results = model_decay.fit(data_dict, num_iterations=200)
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

    # clean up the columns
    col_to_keep = ['participant_id', 'AIC', 'BIC', 't', 'alpha', 'subj_weight']
    dual_results = dual_results[col_to_keep]
    dual_results.rename(columns={'participant_id': 'Subnum'}, inplace=True)

    # merge the data
    data = data.merge(dual_results, on='Subnum')
    data['dist'] = abs(data['best_weight'] - 0.5)

    # save the data
    data.to_csv('./Data/processed_data_modeled.csv', index=False)

    # Now group by trial type and choice to see the difference in anticipatory GSR
    PhasicAnticipatoryGSRAUC_grouped = data.groupby(['Subnum', 'SetSeen.', 'Condition',  'BestOption'])['PhasicAnticipatoryGSRAUC'].mean().reset_index()
    PhasicAnticipatoryGSRAUC_grouped = PhasicAnticipatoryGSRAUC_grouped.pivot(index=['Subnum', 'SetSeen.', 'Condition'], columns='BestOption',
                                      values='PhasicAnticipatoryGSRAUC').reset_index()
    PhasicAnticipatoryGSRAUC_grouped.columns = ['Subnum', 'SetSeen.', 'Condition', 'Optimal', 'Suboptimal']
    CA_AAUC = PhasicAnticipatoryGSRAUC_grouped[PhasicAnticipatoryGSRAUC_grouped['SetSeen.'] == 2]
    CA_AAUC.loc[:, 'GSRdiff'] = CA_AAUC['Suboptimal'] - CA_AAUC['Optimal']

    best_weight_grouped = data.groupby(['Subnum', 'SetSeen.', 'Condition'])['best_weight'].mean().reset_index()
    best_weight_grouped = best_weight_grouped[best_weight_grouped['SetSeen.'] == 2]

    best_prop = data.groupby(['Subnum', 'SetSeen.', 'Condition'])['BestOption'].mean().reset_index()
    best_prop = best_prop[best_prop['SetSeen.'] == 2]



    results = dual_results.merge(CA_AAUC[['Subnum', 'Condition', 'GSRdiff']], on='Subnum')
    results = results.merge(best_weight_grouped [['Subnum', 'best_weight']], on='Subnum')
    results = results.merge(best_prop[['Subnum', 'BestOption']], on='Subnum')
    results = results.dropna()

    results['best_weight'] = pd.to_numeric(results['best_weight'], errors='coerce')

    # pearson r
    print(pearsonr(results['GSRdiff'], results['best_weight']))

    # separate by condition
    baseline = results[results['Condition'] == 'Baseline']
    frequency = results[results['Condition'] == 'Frequency']
    magnitude = results[results['Condition'] == 'Magnitude']

    # t-test
    print(pearsonr(baseline['GSRdiff'], baseline['best_weight']))
    print(pearsonr(frequency['GSRdiff'], frequency['best_weight']))
    print(pearsonr(magnitude['GSRdiff'], magnitude['best_weight']))

    # predict best option by GSR
    model = smf.ols('BestOption ~ Condition + GSRdiff * best_weight', data=results).fit()
    print(model.summary())








