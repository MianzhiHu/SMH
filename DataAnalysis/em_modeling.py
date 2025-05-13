import pandas as pd
from utils.Expectation_Maximization import (em_model, pdf_plot_generator, likelihood_ratio_test,
                                            parameter_extractor, group_assignment)


# Read in the data
data = pd.read_csv('./Data/processed_data_experiment_cvxeda.csv')

CA = data[data['TrialType'] == 2]
CAoptimal = CA.groupby(['Subnum'])['BestOption'].mean().reset_index()
CAoptimal = CAoptimal['BestOption']

# Define model parameters
n_iter = 100

# ======================================================================================================================
# Bimodal Model
# ======================================================================================================================
result_bi = []
for i in range(n_iter):
    print(i)
    mu1, mu2, sd1, sd2, ppi, ll, ll_null, aic, aic_null, bic, bic_null, R2\
        = em_model(CAoptimal, tolerance=1e-10, random_init=True)
    result_bi.append([mu1, mu2, sd1, sd2, ppi, ll, ll_null, aic, aic_null, bic, bic_null, R2])

# convert the result to a dataframe
result_bi = pd.DataFrame(result_bi, columns=['mu1', 'mu2', 'sd1', 'sd2', 'ppi', 'll', 'll_null',
                                       'aic', 'aic_null', 'bic', 'bic_null', 'R2'])
result_bi = result_bi.dropna()
# round the result to 3 decimal places
result_bi = result_bi.round(3)
result_bi.to_csv('./Data/EM/bimodal_CA.csv', index=False)

# ======================================================================================================================
# Trimodal Model
# ======================================================================================================================
result_tri = []
for i in range(n_iter):
    print(i)
    mu1, mu2, mu3, sd1, sd2, sd3, ppi1, ppi2, ppi3, ll, ll_null, aic, aic_null, bic, bic_null, R2\
        = em_model(CAoptimal, tolerance=1e-10, random_init=True, modality='trimodal')

    result_tri.append([mu1, mu2, mu3, sd1, sd2, sd3, ppi1, ppi2, ppi3, ll, ll_null, aic, aic_null, bic, bic_null, R2])

# convert the result to a dataframe
result_tri = pd.DataFrame(result_tri, columns=['mu1', 'mu2', 'mu3', 'sd1', 'sd2', 'sd3', 'ppi1', 'ppi2', 'ppi3', 'll',
                                             'll_null', 'aic', 'aic_null', 'bic', 'bic_null', 'R2'])

# round the result to 3 decimal places
# result_tri = result_tri.dropna()
result_tri = result_tri.round(3)
# result_tri = result_tri.map(lambda x: "{:.3f}".format(x) if isinstance(x, (int, float)) else x)

result_tri.to_csv('./Data/EM/trimodal_CA.csv', index=False)

# ======================================================================================================================
result_bi = pd.read_csv('./Data/EM/bimodal_CA.csv')
result_tri = pd.read_csv('./Data/EM/trimodal_CA.csv')

print(result_tri['mu1'].value_counts())
print(result_tri['mu2'].value_counts())
print(result_tri['mu3'].value_counts())

pdf_plot_generator(CAoptimal, result_tri, './figures/CA_EM', 'trimodal')
pdf_plot_generator(CAoptimal, result_bi, './figures/CA_EM_Bi', 'bimodal')

# likelihood_ratio_test(trimodal_CA, 6)