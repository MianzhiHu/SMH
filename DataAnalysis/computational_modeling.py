import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.ComputationalModeling import ComputationalModels, dict_generator
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

dual_results = model_dual.fit(data_dict, 'Entropy_Dis_ID', Gau_fun='Naive_Recency', Dir_fun='Linear_Recency',
                              weight_Dir='softmax', weight_Gau='softmax', num_iterations=200)
dual_results.to_csv('./models/dual_results.csv', index=False)