import numpy as np
import pandas as pd
import ast
import json
import neurokit2 as nk
from sklearn.model_selection import learning_curve
from sympy.codegen.ast import continue_
from utilities.utility_processGSR import (extract_samples, processGSR, rename_columns, interleave_columns,
                                          difference_transformation, unzip_combined_data, area_under_curve,
                                          check_best_option, cda_pipeline, combined_cda_pipeline, difference_pipeline,
                                          auto_pipeline, neurokit_pipeline, method_pipeline, find_peak, trialwise,
                                          trial_split, unit_mapping, calculate_r_squared_reconstruction, calculate_snr,
                                          signed_log_transform)
from scipy.signal import correlate
from utilities.pyEDA.main import *
import pyphysio as ph
from utils.dfa import *
from pyphysio.specialized.eda import DriverEstim, PhasicEstim
from MFDFA import MFDFA

# ======================================================================================================================
# Load data from JATOS generated file
# ======================================================================================================================
# # parse the data
# data = []
#
# with open('./Data/jatos_results_data_20241114213941.txt', 'r') as file:
#     for line in file:
#         json_data = json.loads(line)
#         data.append(json_data)
#
# data = pd.DataFrame(data)

# Specify the folder containing the .txt files
directory_path = './Data/E2'

# Extract experiment ID from path
ex_id = directory_path.split('/')[-1]
# directory_path = './Data/Test Data'

# Define parameters for the GSR data processing
sample_rate = 100
interval = 10  # interval only for the difference transformation method
amplitude_threshold = 0.01  # amplitude threshold to detect SCR
iterations = 0
max_gsr_samples = 800
max_trial_samples = 250 * 2
learning_threshold = 0.55
ts_option = 'experiment'  # choose from 'experiment', 'trial', or 'trial_split'
method = 'cvxeda'  # choose from 'auto', 'highpass', 'smoothmedian', 'cvxeda', 'cda', 'SparsEDA', 'difference', or 'combined'
z_score = False

# Initialize a list to store all data
all_data = []

# Iterate over each .txt file in the directory and read the contents
for filename in os.listdir(directory_path):
    if filename.endswith('.txt'):
        file_path = os.path.join(directory_path, filename)
        print(f"Reading file: {file_path}")
        with open(file_path, 'r') as file:
            for line in file:
                json_data = json.loads(line)
                all_data.append(json_data)

# Combine all data into a single DataFrame
data = pd.DataFrame(all_data)

# ======================================================================================================================
# Preprocess behavioral data
# ======================================================================================================================

# Categorize variables in the data
behavioral_list = ['ReactTime', 'Reward', 'BestOption', 'KeyResponse', 'SetSeen ', 'OptionRwdMean', 'Phase']

GSR_list = ['AnticipatoryGSR', 'OutcomeGSR']

knowledge_list = ['OptionOrder', 'EstOptionA', 'OptionAConfidence', 'EstOptionS', 'OptionSConfidence',
                  'EstOptionK', 'OptionKConfidence', 'EstOptionL', 'OptionLConfidence']

personality_list = ['BDI_Total', 'BAI_Total', 'BISScore', 'PSWQScore', 'ESIBF_disinhScore', 'ESIBF_aggreScore',
                    'ESIBF_sScore']

demo_list = ['Gender', 'Ethnicity', 'Race', 'Age']


if ex_id == 'E2':
    # add the high stakes column
    behavioral_list += ['HighStakes']

kept_columns = (['Condition'] + ['studyResultId'] + behavioral_list + GSR_list +
                personality_list + demo_list)

# keep only the necessary columns
data = data[kept_columns]

# refine the data
df = data.groupby(data.index // 10).first()
df.reset_index(drop=True, inplace=True)

# check the number of GSR data points
print(df['AnticipatoryGSR'].apply(lambda x: len([lst for lst in x if len(lst) > 0])).value_counts())
print()
print(f'Missing data percentage in AnticipatoryGSR: '
      f'{df['AnticipatoryGSR'].apply(lambda x: len([lst for lst in x if len(lst) > 0]) != 250).mean() * 100:.2f}%')
print()
print(df['OutcomeGSR'].apply(lambda x: len([lst for lst in x if len(lst) > 0])).value_counts())
print()
print(f'Missing data percentage in OutcomeGSR: '
      f'{(df['OutcomeGSR'].apply(lambda x: len([lst for lst in x if len(lst) > 0]) != 250)).mean() * 100:.2f}%')

# remove data where either anticipatory or outcome GSR has less than 250 data points
df = df[(df['AnticipatoryGSR'].apply(lambda x: len([lst for lst in x if len(lst) > 0])) == 250) &
        (df['OutcomeGSR'].apply(lambda x: len([lst for lst in x if len(lst) > 0])) == 250)]
print(f'Remaining data: {len(df)}')

# explode the data
df = df.explode(behavioral_list + GSR_list)

# extract the numbers from lists
df[personality_list] = df[personality_list].map(lambda x: x[0] if x else np.nan)

# extract the demographic data from dictionary-like strings
df[demo_list] = df[demo_list].map(lambda x: ast.literal_eval(x).get('Q0', None) if x else np.nan)

# extract the GSR data
df[GSR_list] = df[GSR_list].map(extract_samples)

# calculate the area under the curve for each GSR data
df[GSR_list] = df[GSR_list].map(str)
ts_ant_gsr, ts_out_gsr = processGSR(df)

# # pick the first column
# gsr_test = preprocessed
# gsr_test = np.array(ts_ant_gsr.iloc[:, 0])
# gsr_test_reshaped = gsr_test.reshape(-1, 1)
# min_window = 200
# max_window = 13 * 100
# q_val = np.array([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5])
# lag = np.arange(min_window, max_window + 1, 1)
# lag, dfa_results = MFDFA(gsr_test, lag = lag, q=q_val, order=1)
# H_hat = np.polyfit(np.log(lag),np.log(dfa_results),1)[0]
# hurst, cis, rsquared = dfa(gsr_test_reshaped, max_window_size=max_window, min_window_size=min_window, return_confidence_interval=True)
# print(f'H using dfa: {hurst}; H using MFDFA: {H_hat}')

# ======================================================================================================================
# Preprocess the GSR data
# ======================================================================================================================
# Rename the columns to include the participant number and trial number
ts_ant_gsr_renamed = rename_columns(ts_ant_gsr.copy())
ts_out_gsr_renamed = rename_columns(ts_out_gsr.copy())

# print mean anticipatory and outcome GSR data length
print(f'Mean count of the data: {ts_ant_gsr.count().mean()}')
print(f'Mean count of the data: {ts_out_gsr.count().mean()}')

# Get unique participants from the columns
participants = ts_ant_gsr.columns.unique()

print()
print('=========================================================================================================')
print(f'Processing GSR data using the {method} method...')
print(f'Data will be processed on a {ts_option} basis.')
print('=========================================================================================================')
print()

# Determine whether we want to combine all signals into a single time series
if ts_option == 'experiment':

    # Interleave the columns from both DataFrames
    combined_df = interleave_columns(ts_ant_gsr_renamed, ts_out_gsr_renamed)

    # Initialize the data dictionaries for combined results
    combined_cleaned_gsr = {}
    combined_phasic_gsr = {}
    combined_tonic_gsr = {}

    # Process the GSR data for each participant
    for participant in participants:

        iterations += 1
        print(f'Processing participant ({iterations}/{len(participants)})...')

        participant_cols_full = [col for col in combined_df.columns if col.startswith(f"{participant}/")]
        # drop duplicates in the columns
        participant_cols = list(dict.fromkeys(participant_cols_full))

        # Combine the data for the participant
        gsr_data = combined_df[participant_cols]

        # Flatten the data
        data_flat = gsr_data.values.flatten()

        # Remove NaN values
        data_flat = data_flat[~np.isnan(data_flat)]

        # Apply the preprocessing function
        preprocessed = nk.signal_sanitize(nk.eda_clean(data_flat, sampling_rate=sample_rate, method='BioSPPy'))

        # Apply the method pipeline
        phasic, tonic = method_pipeline[method](preprocessed, sample_rate, interval, amplitude_threshold, method)

        # Record the number of valid data points in each column
        valid_data = gsr_data.count()

        # Reverse back to the original shape column-wise using the number of valid data points
        for j, signals in enumerate([preprocessed, phasic, tonic]):
            # check if signal contains negative values
            if np.min(signals) < 0:
                if j == 0:
                    print(f'[{method}]: Participant {iterations} has negative values in the original signal.')
                elif j == 1:
                    print(f'[{method}]: Participant {iterations} has negative values in the phasic signal.')
                else:
                    print(f'[{method}]: Participant {iterations} has negative values in the tonic signal.')
            else:
                pass

            # Split the signals based on the number of valid data points
            signals_split = np.split(signals, np.cumsum(valid_data), axis=0)[:-1]
            signals_df = [pd.DataFrame(k).reset_index(drop=True) for k in signals_split]
            signals_combined = pd.concat(signals_df, axis=1)

            # Set column names for the signals_combined DataFrame
            signals_combined.columns = participant_cols_full

            if j == 0:
                combined_cleaned_gsr[participant] = signals_combined
            elif j == 1:
                combined_phasic_gsr[participant] = signals_combined
            else:
                combined_tonic_gsr[participant] = signals_combined

    # Concatenate results for each type across all participants
    cleaned_combined_gsr = pd.concat(combined_cleaned_gsr.values(), axis=1)
    phasic_combined_gsr = pd.concat(combined_phasic_gsr.values(), axis=1)
    tonic_combined_gsr = pd.concat(combined_tonic_gsr.values(), axis=1)

    if (cleaned_combined_gsr.shape[0] == phasic_combined_gsr.shape[0] == tonic_combined_gsr.shape[0] == max_gsr_samples
            and cleaned_combined_gsr.shape[1] == phasic_combined_gsr.shape[1] == tonic_combined_gsr.shape[1]
            == len(participants) * max_trial_samples):
        pass
    else:
        raise ValueError('The data has not been processed correctly.')

    # Unzip the combined data
    cleaned_ant_gsr, cleaned_out_gsr = unzip_combined_data(cleaned_combined_gsr)
    phasic_ant_gsr, phasic_out_gsr = unzip_combined_data(phasic_combined_gsr)
    tonic_ant_gsr, tonic_out_gsr = unzip_combined_data(tonic_combined_gsr)

# Or, we can preprocess each time series individually for each trial
else:
    cleaned_ant_gsr_list = []
    cleaned_out_gsr_list = []
    phasic_ant_gsr_list = []
    phasic_out_gsr_list = []
    tonic_ant_gsr_list = []
    tonic_out_gsr_list = []

    for i in range(ts_ant_gsr_renamed.shape[1]):

        if i % 250 == 0:
            print(f'Processing participant ({i // 250 + 1}/{len(participants)})...')

        ant_gsr = ts_ant_gsr_renamed.iloc[:, i]
        ant_gsr = ant_gsr[~np.isnan(ant_gsr)]
        out_gsr = ts_out_gsr_renamed.iloc[:, i]
        out_gsr = out_gsr[~np.isnan(out_gsr)]

        # if the data is too short, put as NaN
        if (len(ant_gsr) < 300) or (len(out_gsr) < 300):
            cleaned_ant_gsr_list.append(pd.Series(np.nan, index=np.arange(max_gsr_samples)))
            cleaned_out_gsr_list.append(pd.Series(np.nan, index=np.arange(max_gsr_samples)))
            phasic_ant_gsr_list.append(pd.Series(np.nan, index=np.arange(max_gsr_samples)))
            phasic_out_gsr_list.append(pd.Series(np.nan, index=np.arange(max_gsr_samples)))
            tonic_ant_gsr_list.append(pd.Series(np.nan, index=np.arange(max_gsr_samples)))
            tonic_out_gsr_list.append(pd.Series(np.nan, index=np.arange(max_gsr_samples)))
            continue

        # preprocess
        cleaned_ant_gsr, cleaned_out_gsr, phasic_ant_gsr, phasic_out_gsr, tonic_ant_gsr, tonic_out_gsr = (
            unit_mapping[ts_option](ant_gsr, out_gsr, sample_rate, interval, amplitude_threshold,
                                    method, max_gsr_samples))

        # append the data
        cleaned_ant_gsr_list.append(cleaned_ant_gsr)
        cleaned_out_gsr_list.append(cleaned_out_gsr)
        phasic_ant_gsr_list.append(phasic_ant_gsr)
        phasic_out_gsr_list.append(phasic_out_gsr)
        tonic_ant_gsr_list.append(tonic_ant_gsr)
        tonic_out_gsr_list.append(tonic_out_gsr)

    cleaned_ant_gsr = pd.DataFrame(cleaned_ant_gsr_list).transpose()
    cleaned_out_gsr = pd.DataFrame(cleaned_out_gsr_list).transpose()
    phasic_ant_gsr = pd.DataFrame([item.values for item in phasic_ant_gsr_list]).transpose()
    phasic_out_gsr = pd.DataFrame([item.values for item in phasic_out_gsr_list]).transpose()
    tonic_ant_gsr = pd.DataFrame([item.values for item in tonic_ant_gsr_list]).transpose()
    tonic_out_gsr = pd.DataFrame([item.values for item in tonic_out_gsr_list]).transpose()

    cleaned_ant_gsr.columns = ts_ant_gsr_renamed.columns
    cleaned_out_gsr.columns = ts_out_gsr_renamed.columns
    phasic_ant_gsr.columns = ts_ant_gsr_renamed.columns
    phasic_out_gsr.columns = ts_out_gsr_renamed.columns
    tonic_ant_gsr.columns = ts_ant_gsr_renamed.columns
    tonic_out_gsr.columns = ts_out_gsr_renamed.columns

print('=========================================================================================================')
print('GSR data has been successfully processed. Now quality checking...')
print('=========================================================================================================')

# ======================================================================================================================
# Quality check
# ======================================================================================================================
# Check the quality of the reconstruction
r_squared_ant = []
r_squared_out = []

for i in range(phasic_ant_gsr.shape[1]):
    original = cleaned_ant_gsr.iloc[:, i]
    tonic = tonic_ant_gsr.iloc[:, i]
    phasic = phasic_ant_gsr.iloc[:, i]
    r_squared_ant.append(calculate_r_squared_reconstruction(original, tonic, phasic))

    original = cleaned_out_gsr.iloc[:, i]
    tonic = tonic_out_gsr.iloc[:, i]
    phasic = phasic_out_gsr.iloc[:, i]
    r_squared_out.append(calculate_r_squared_reconstruction(original, tonic, phasic))

for r_squared_list, name in zip([r_squared_ant, r_squared_out], ['anticipatory', 'outcome']):
    print(f'[{method}] By reconstructing the phasic and tonic components in the {name} GSR data, '
            f'the mean R-squared value is {np.nanmean(r_squared_list):.2f} '
          f'with a standard deviation of {np.nanstd(r_squared_list):.2f}.')
    print(f'[{method}] The percentage of data with R-squared value between 0 and 1 is '
            f'{len([r for r in r_squared_list if 0 < r < 1]) / len(r_squared_list) * 100:.2f}%; and '
            f'{len([r for r in r_squared_list if 0.85 < r < 1]) / len(r_squared_list) * 100:.2f}% are above 0.85.')
    print(f'[{method}] The minimum R-squared value is {min(r_squared_list):.2f} '
          f'and the maximum R-squared value is {max(r_squared_list):.2f}.')
    print()

print(f'=========================================================================================================')

# Calculate the signal-to-noise ratio for the phasic and tonic components
snr_ant_tonic = []
snr_ant_phasic = []
snr_out_tonic = []
snr_out_phasic = []

for i in range(phasic_ant_gsr.shape[1]):
    tonic_ant_noise = cleaned_ant_gsr.iloc[:, i] - tonic_ant_gsr.iloc[:, i]
    phasic_ant_noise = cleaned_ant_gsr.iloc[:, i] - phasic_ant_gsr.iloc[:, i]
    tonic_out_noise = cleaned_out_gsr.iloc[:, i] - tonic_out_gsr.iloc[:, i]
    phasic_out_noise = cleaned_out_gsr.iloc[:, i] - phasic_out_gsr.iloc[:, i]

    snr_ant_tonic.append(calculate_snr(tonic_ant_gsr.iloc[:, i], tonic_ant_noise))
    snr_ant_phasic.append(calculate_snr(phasic_ant_gsr.iloc[:, i], phasic_ant_noise))
    snr_out_tonic.append(calculate_snr(tonic_out_gsr.iloc[:, i], tonic_out_noise))
    snr_out_phasic.append(calculate_snr(phasic_out_gsr.iloc[:, i], phasic_out_noise))

for snr_list, name in zip([snr_ant_tonic, snr_ant_phasic, snr_out_tonic, snr_out_phasic],
                            ['anticipatory tonic', 'anticipatory phasic', 'outcome tonic', 'outcome phasic']):
    print(f'[{method}] The mean signal-to-noise ratio for the {name} component is {np.nanmean(snr_list):.2f} dB '
            f'with a standard deviation of {np.nanstd(snr_list):.2f} dB.')
    print(f'[{method}] The minimum signal-to-noise ratio is {min(snr_list):.2f} dB '
            f'and the maximum signal-to-noise ratio is {max(snr_list):.2f} dB.')
    print(f'[{method}] The percentage of data with signal-to-noise ratio above 5 dB is '
            f'{len([snr for snr in snr_list if snr > 5]) / len(snr_list) * 100:.2f}%.')
    print(f'[{method}] The percentage of data with signal-to-noise ratio above 10 dB is '
            f'{len([snr for snr in snr_list if snr > 10]) / len(snr_list) * 100:.2f}%.')
    print(f'[{method}] The percentage of data with signal-to-noise ratio above 15 dB is '
            f'{len([snr for snr in snr_list if snr > 15]) / len(snr_list) * 100:.2f}%.')
    print(f'[{method}] The percentage of data with signal-to-noise ratio above 20 dB is '
            f'{len([snr for snr in snr_list if snr > 20]) / len(snr_list) * 100:.2f}%.')
    print()
print(f'=========================================================================================================')

# ======================================================================================================================
# Calculate the area under the curve for each GSR data
df['AnticipatoryGSRAUC'] = area_under_curve(cleaned_ant_gsr)
df['OutcomeGSRAUC'] = area_under_curve(cleaned_out_gsr)
df['TonicAnticipatoryGSRAUC'] = area_under_curve(tonic_ant_gsr)
df['PhasicAnticipatoryGSRAUC'] = area_under_curve(phasic_ant_gsr)
df['PhasicAnticipatoryGSRPeak'] = find_peak(phasic_ant_gsr)
df['TonicOutcomeGSRAUC'] = area_under_curve(tonic_out_gsr)
df['PhasicOutcomeGSRAUC'] = area_under_curve(phasic_out_gsr)
df['PhasicOutcomeGSRPeak'] = find_peak(phasic_out_gsr)

# log-transform the GSR data to remove skewness
auc_columns = ['AnticipatoryGSRAUC', 'OutcomeGSRAUC', 'TonicAnticipatoryGSRAUC', 'PhasicAnticipatoryGSRAUC',
               'TonicOutcomeGSRAUC', 'PhasicOutcomeGSRAUC']
df[auc_columns] = df[auc_columns].apply(lambda x: signed_log_transform(x))

# calculate additional GSR features
df['GSRAUC'] = df['AnticipatoryGSRAUC'] + df['OutcomeGSRAUC']
df['TonicGSRAUC'] = df['TonicAnticipatoryGSRAUC'] + df['TonicOutcomeGSRAUC']
df['PhasicGSRAUC'] = df['PhasicAnticipatoryGSRAUC'] + df['PhasicOutcomeGSRAUC']

# take the z-score of the GSR data withing each participant
if z_score is True:
    all_auc_columns = ['AnticipatoryGSRAUC', 'OutcomeGSRAUC', 'TonicAnticipatoryGSRAUC', 'PhasicAnticipatoryGSRAUC',
                       'TonicOutcomeGSRAUC', 'PhasicOutcomeGSRAUC', 'GSRAUC', 'TonicGSRAUC', 'PhasicGSRAUC']
    for auc_column in all_auc_columns:
        df[auc_column] = df.groupby('studyResultId')[auc_column].transform(lambda x: (x - x.mean()) / x.std())

# # Calculate Hurst exponent
# hurst = []
# r2 = []
# min_w = 50
# for i in range(phasic_ant_gsr.shape[1]):
#     selected = phasic_ant_gsr.to_numpy()[:, i]
#     selected_cleaned = selected[~np.isnan(selected)].reshape(-1, 1)
#     max_w = 200
#     h, _, r = dfa(selected_cleaned, max_window_size=max_w, min_window_size=min_w, return_confidence_interval=True)
#     hurst.append(h[0])
#     r2.append(r[0])
#
# print(f'Max Hurst exponent: {max(hurst)}; Min Hurst exponent: {min(hurst)}')
# print(f'Percentage of Hurst exponent between 0 and 1: {len([h for h in hurst if 0 < h < 1]) / len(hurst) * 100:.2f}%')
# # print the r-squared values for those Hurst not between 0 and 1
# # record the indices of the Hurst exponent that are not between 0 and 1
# indices = [i for i, h in enumerate(hurst) if h < 0 or h > 1]
# # use these indices to print all the r-squared values
# print(f'R-squared values for Hurst exponent not between 0 and 1: {np.array(r2)[indices]}')
#
# df['Hurst'] = hurst


# ======================================================================================================================
# Some final cleaning steps
# ======================================================================================================================

# add a participant ID every 250 rows
subject_id = len(df) // 250 + 1
ids = np.arange(1, subject_id)
ids = np.repeat(ids, 250)
df['studyResultId'] = ids
df = df.rename(columns={'studyResultId': 'Subnum'})
col = df.pop('Subnum')
df.insert(0, 'Subnum', col)

# add a trial index column from 1 to 250
df['Trial_Index'] = np.tile(np.arange(1, 251), len(df) // 250)
col = df.pop('Trial_Index')
df.insert(1, 'Trial_Index', col)

# change the phase column so that it separates every 25 trials
conditions = [
    (df['Trial_Index'] <= 25),
    (df['Trial_Index'] <= 50),
    (df['Trial_Index'] <= 75),
    (df['Trial_Index'] <= 100),
    (df['Trial_Index'] <= 125),
    (df['Trial_Index'] <= 150),
    (df['Trial_Index'] <= 250),
]

# The corresponding phases for each condition
phases = ['Phase 1', 'Phase 2', 'Phase 3', 'Phase 4', 'Phase 5', 'Phase 6', 'Test']

# Applying the conditions and phases
df['Phase'] = np.select(conditions, phases)

# Finally, create a mapping to check for the best option column
best_option = {
    0: 1,  # choose A in AB
    1: 3,  # choose C in CD
    2: 3,  # choose C in CA
    3: 3,  # choose C in CB
    4: 1,  # choose A in AD
    5: 2  # choose B in BD
}

# Apply the mapping to the BestOption column
df['BestOptionChosen'] = df.apply(check_best_option, mapping=best_option, axis=1)

# check if this new column is exactly the same as the original BestOption column
if (df['BestOptionChosen'] == df['BestOption']).all():
    print('The BestOption column is correct.')
    # remove the new column
    df.drop(columns=['BestOptionChosen'], inplace=True)
else:
    print('The BestOption column is incorrect. Just replaced it.')
    # replace the original column with the new one
    df['BestOption'] = df['BestOptionChosen']
    df.drop(columns=['BestOptionChosen'], inplace=True)

# remove the original GSR data
df.drop(columns=['AnticipatoryGSR', 'OutcomeGSR'], inplace=True)

# rename the columns
df.rename(columns={'SetSeen ': 'TrialType'}, inplace=True)

# save the data
df.to_csv(f'./Data/{ex_id}/Preprocessed/processed_data_{ts_option}_{method}_{ex_id}.csv', index=False)

# save good learner data separately
training_data = df[df['Phase'] != 'Test']
training_accuracy = training_data.groupby(['Subnum'])['BestOption'].mean().reset_index()
good_learner_id = training_accuracy[training_accuracy['BestOption'] >= learning_threshold]['Subnum']
good_learner_data = df[df['Subnum'].isin(good_learner_id)]
np.save(f'./Data/{ex_id}/Preprocessed/good_learner_id_{ex_id}.npy', good_learner_id.to_numpy())
good_learner_data.to_csv(f'./Data/{ex_id}/Preprocessed/good_learner_data_{ts_option}_{method}_{ex_id}.csv', index=False)
print(f'There are {len(training_accuracy[training_accuracy["BestOption"] < learning_threshold])} bad learners. '
      f'{len(training_accuracy[training_accuracy["BestOption"] >= learning_threshold])} good learners are retained.')

print('=========================================================================================================')
print('Done! Preprocessed data has been saved!')
print('=========================================================================================================')

# ======================================================================================================================
# for illustration, we will use the average of the anticipatory GSR data to show the preprocessing steps
# ======================================================================================================================
# # preprocess the data
# original = ts_ant_gsr.mean(axis=1)[0:450]
#
# cleaned = nk.eda_clean(original, sampling_rate=100)
# signals = nk.eda_phasic(cleaned, sampling_rate=100)
#
#
# tonic_gsr = signals['EDA_Tonic']
# phasic_gsr = signals['EDA_Phasic']
#
#
# cleaned = pd.DataFrame(cleaned)
# cleaned.index = cleaned.index * 10
# tonic_gsr = pd.DataFrame(tonic_gsr)
# tonic_gsr.index = tonic_gsr.index * 10
# phasic_gsr = pd.DataFrame(phasic_gsr)
# phasic_gsr.index = phasic_gsr.index * 10
#
#
# # Plot the original and filtered signals
# palette = sns.color_palette("husl", 3)
# sns.set_style("white")
# plt.figure()
# plt.plot(original, color=palette[0], label='Original Signal')
# plt.tight_layout()
# sns.despine()
# plt.savefig('./figures/preprocessing_original.png', dpi=300)
# plt.show()
#
# plt.figure()
# plt.plot(tonic_gsr, color=palette[1], label='Tonic GSR')
# plt.tight_layout()
# sns.despine()
# plt.savefig('./figures/preprocessing_tonic.png', dpi=300)
# plt.show()
#
# plt.figure()
# plt.plot(phasic_gsr, color=palette[2], label='Phasic GSR')
# plt.tight_layout()
# sns.despine()
# plt.savefig('./figures/preprocessing_phasic.png', dpi=300)
# plt.show()
#
#
# # plot together
# plt.figure()
# plt.plot(original, color=palette[0], label='Original Signal')
# plt.plot(cleaned, color=palette[1], label='Filtered Signal')
# plt.legend()
# plt.tight_layout()
# sns.despine()
# plt.savefig('./figures/preprocessing_combined.png', dpi=300)
# plt.show()

# # plot the phasic and tonic GSR data
# plt.figure()
# plt.plot(phasic_ant_gsr.mean(axis=1), label='Tonic Anticipatory GSR')
# plt.xlabel('Time (ms)')
# plt.ylabel('GSR')
# plt.legend()
# plt.show()


# # Simulate EDA signal
# eda_signal = nk.eda_simulate(duration=30, scr_number=3, drift=0.3, sampling_rate=sample_rate, noise=0.04)
# eda_signal = nk.standardize(eda_signal)
# eda_signal = nk.eda_clean(eda_signal, sampling_rate=sample_rate)
#
# # Decompose using different algorithms
# difference = pd.DataFrame(difference_transformation(eda_signal, interval=10, sample_rate=sample_rate))
# smoothMedian = nk.eda_phasic(eda_signal, method='smoothmedian', sampling_rate=sample_rate)
# highpass = nk.eda_phasic(eda_signal, method='highpass', sampling_rate=sample_rate)
# cvx = nk.eda_phasic(eda_signal, method='cvxeda', sampling_rate=sample_rate)
# # sparse = nk.eda_phasic(eda_signal, method='sparse', sampling_rate=sample_rate)
#
# cda_sig = ph.create_signal(eda_signal, sampling_freq=sample_rate)
# cda_driver = DriverEstim()(cda_sig)
# cda_phasic = PhasicEstim(amplitude=0.1)(cda_driver)
# cda = cda_phasic.to_dataframe()['signal_DriverEstim_PhasicEstim']
#
# # Extract tonic and phasic components for plotting
# p1 = difference.values.reshape(-1)
# p2 = highpass["EDA_Phasic"].values
# p3 = smoothMedian["EDA_Phasic"].values
# p4 = cda.values
# # replace values higher than 10 with average without them
# p4[p4 > 10] = p4[p4 <= 10].mean()
# p5 = cvx["EDA_Phasic"].values
# # p6 = sparse["EDA_Phasic"].values
#
# nk.signal_plot([eda_signal, p1, p2, p3, p5], labels=["Original", "Difference", "Highpass", "SmoothMedian",
#                                                      "CVX"])
# plt.show(dpi=600)
#
# nk.signal_plot([eda_signal, p4], labels=["Original", "CDA"])
# plt.show(dpi=600)