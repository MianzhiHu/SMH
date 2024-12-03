import pandas as pd
import ast
import json
import neurokit2 as nk
from utilities.utility_processGSR import (extract_samples, processGSR, rename_columns, interleave_columns,
                                          difference_transformation, unzip_combined_data, area_under_curve,
                                          check_best_option)
from utilities.pyEDA.main import *
import pyphysio as ph
from pyphysio.specialized.eda import DriverEstim, PhasicEstim

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
directory_path = './Data'

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

# ======================================================================================================================
# Preprocess the GSR data
# ======================================================================================================================

# Rename the columns to include the participant number and trial number
ts_ant_gsr_renamed = rename_columns(ts_ant_gsr.copy())
ts_out_gsr_renamed = rename_columns(ts_out_gsr.copy())

# print mean anticipatory and outcome GSR data length
print(f'Mean count of the data: {ts_ant_gsr.count().mean()}')
print(f'Mean count of the data: {ts_out_gsr.count().mean()}')

# Interleave the columns from both DataFrames
combined_df = interleave_columns(ts_ant_gsr_renamed, ts_out_gsr_renamed)

# standardize the data using log transformation
combined_df = np.log(combined_df + 1)

# Initialize the data dictionaries for combined results
combined_cleaned_gsr = {}
combined_phasic_gsr = {}
combined_tonic_gsr = {}

# Get unique participants from the columns
participants = ts_ant_gsr.columns.unique()

iterations = 0
method = 'cda'  # choose from 'highpass', 'smoothmedian', 'cvxeda', 'cda', 'sparse', and 'difference'
print()
print('=========================================================================================================')
print(f'Processing GSR data using the {method} method...')
print('=========================================================================================================')
print()

# Process the GSR data for each participant
for participant in participants:

    sample_rate = 100
    interval = 10  # interval only for the difference transformation method
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
    preprocessed = nk.signal_sanitize(nk.eda_clean(data_flat, sampling_rate=sample_rate, method='biosppy'))

    # ------------------------------------------------------------------------------------------------------------------
    # NeuroKit2 can implement the three methods for phasic and tonic decomposition: high-pass filter,
    # median smoothing,
    # sparse deconvolution,
    # and convex optimization.
    # The change in the method can be done by simply changing the method parameter in the
    # eda_phasic function.
    # The default method is the high-pass filter.
    # ------------------------------------------------------------------------------------------------------------------
    if method in ['highpass', 'smoothmedian', 'cvxeda', 'sparse']:
        sig = nk.eda_phasic(preprocessed, sampling_rate=sample_rate, method=method)

        phasic = sig['EDA_Phasic']
        tonic = sig['EDA_Tonic']

    # ------------------------------------------------------------------------------------------------------------------
    # However, the continuous deconvolution analysis (CDA) can only be implemented using the pyphysio library.
    # ------------------------------------------------------------------------------------------------------------------
    if method == 'cda':
        # create the data for the driver
        sig = ph.create_signal(preprocessed, sampling_freq=sample_rate)

        driver = DriverEstim()(sig)
        phasic_sig = PhasicEstim(amplitude=0.01)(driver)
        tonic_sig = PhasicEstim(amplitude=0.01, return_phasic=False)(driver)

        # revert back to the original shape
        phasic = phasic_sig.to_dataframe()['signal_DriverEstim_PhasicEstim']
        tonic = tonic_sig.to_dataframe()['signal_DriverEstim_PhasicEstim']

    # ------------------------------------------------------------------------------------------------------------------
    # The original preprocessing method by Bechara et al., 1999, uses the difference transformation method. There is
    # no existing package in Python that implements this method. Therefore, we custom implement it here.
    # ------------------------------------------------------------------------------------------------------------------
    if method == 'difference':
        phasic = pd.DataFrame(difference_transformation(preprocessed, interval=interval, sample_rate=sample_rate))
        # there is no tonic component in the difference transformation method
        tonic = pd.DataFrame(np.zeros(len(phasic)))

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

        # # standardize the data with the log transformation
        # signals = np.log(signals + 1)
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

# check the shape of the data
max_gsr_samples = 800
max_trial_samples = 250 * 2

if (cleaned_combined_gsr.shape[0] == phasic_combined_gsr.shape[0] == tonic_combined_gsr.shape[0] == max_gsr_samples
        and cleaned_combined_gsr.shape[1] == phasic_combined_gsr.shape[1] == tonic_combined_gsr.shape[1]
        == len(participants) * max_trial_samples):
    print('=========================================================================================================')
    print('GSR data has been successfully processed.')
    print('=========================================================================================================')
else:
    raise ValueError('The data has not been processed correctly.')

# Unzip the combined data
cleaned_ant_gsr, cleaned_out_gsr = unzip_combined_data(cleaned_combined_gsr)
phasic_ant_gsr, phasic_out_gsr = unzip_combined_data(phasic_combined_gsr)
tonic_ant_gsr, tonic_out_gsr = unzip_combined_data(tonic_combined_gsr)

# Calculate the area under the curve for each GSR data
df['AnticipatoryGSRAUC'] = area_under_curve(cleaned_ant_gsr)
df['OutcomeGSRAUC'] = area_under_curve(cleaned_out_gsr)
df['TonicAnticipatoryGSRAUC'] = area_under_curve(tonic_ant_gsr)
df['PhasicAnticipatoryGSRAUC'] = area_under_curve(phasic_ant_gsr)
df['TonicOutcomeGSRAUC'] = area_under_curve(tonic_out_gsr)
df['PhasicOutcomeGSRAUC'] = area_under_curve(phasic_out_gsr)
df['GSRAUC'] = df['AnticipatoryGSRAUC'] + df['OutcomeGSRAUC']
df['TonicGSRAUC'] = df['TonicAnticipatoryGSRAUC'] + df['TonicOutcomeGSRAUC']
df['PhasicGSRAUC'] = df['PhasicAnticipatoryGSRAUC'] + df['PhasicOutcomeGSRAUC']

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

# give the average tonic GSR data
df['AverageAnticipatoryTonicAUC'] = df.groupby('Subnum')['TonicAnticipatoryGSRAUC'].transform('mean')
df['AverageOutcomeTonicAUC'] = df.groupby('Subnum')['TonicOutcomeGSRAUC'].transform('mean')

# save the data
df.to_csv(f'./Data/processed_data_{method}.csv', index=False)

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