import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.ComputationalModeling import ComputationalModels, dict_generator, parameter_extractor, trial_exploder
from utils.DualProcess import DualProcessModel
from scipy.stats import pearsonr, spearmanr, ttest_ind

# load processed data
data = pd.read_csv('./Data/processed_data_cda.csv')

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

    # # save the data
    # data.to_csv('./Data/processed_data_cda_modeled.csv', index=False)








