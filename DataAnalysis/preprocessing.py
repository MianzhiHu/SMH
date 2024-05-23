import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import requests
import seaborn as sns
import json
import neurokit2 as nk
import io
import contextlib
from utilities.utility_processGSR import processGSR, area_under_curve, extract_samples, check_best_option
from scipy.signal import butter, filtfilt, welch
from utilities.pyEDA.main import *


# parse the data
data = []

with open('./Data/jatos_results_20240422185445.txt', 'r') as file:
    for line in file:
        json_data = json.loads(line)
        data.append(json_data)

data = pd.DataFrame(data)

# here is where we load and preprocess the data
behavioral_list = ['ReactTime', 'Reward', 'BestOption', 'KeyResponse', 'SetSeen ', 'OptionRwdMean', 'Phase']

GSR_list = ['AnticipatoryGSR', 'OutcomeGSR']

knowledge_list = ['OptionOrder', 'EstOptionA', 'OptionAConfidence', 'EstOptionS', 'OptionSConfidence',
                  'EstOptionK', 'OptionKConfidence', 'EstOptionL', 'OptionLConfidence']

personality_list = ['BDI_Total', 'BAI_Total', 'BISScore', 'PSWQScore', 'ESIBF_disinhScore', 'ESIBF_aggreScore',
                    'ESIBF_sScore']

demo_list = ['Gender', 'Ethnicity', 'Race', 'Age']

kept_columns = (['Condition'] + ['studyResultId'] + behavioral_list + GSR_list +
                personality_list + demo_list)

data = data[kept_columns]

df = data.groupby(data.index // 10).first()
df.reset_index(drop=True, inplace=True)

# check the number of GSR data points
print(df['AnticipatoryGSR'].apply(lambda x: len(x)).value_counts())
print(f'Missing data percentage in AnticipatoryGSR: '
      f'{(1 - df["AnticipatoryGSR"].apply(lambda x: len(x)).value_counts()[250] / len(df)) * 100:.2f}%')
print(df['OutcomeGSR'].apply(lambda x: len(x)).value_counts())
print(f'Missing data percentage in OutcomeGSR: '
      f'{(1 - df["OutcomeGSR"].apply(lambda x: len(x)).value_counts()[250] / len(df)) * 100:.2f}%')

# remove data where either anticipatory or outcome GSR has less than 250 data points
df = df[(df['AnticipatoryGSR'].apply(lambda x: len(x)) == 250) & (df['OutcomeGSR'].apply(lambda x: len(x)) == 250)]

# explode the data
df = df.explode(behavioral_list + GSR_list)

# extract the numbers from lists
df[personality_list] = df[personality_list].applymap(lambda x: x[0] if x else np.nan)

# extract the demographic data from dictionary-like strings
df[demo_list] = df[demo_list].applymap(lambda x: ast.literal_eval(x).get('Q0', None) if x else np.nan)

# extract the GSR data
df[GSR_list] = df[GSR_list].applymap(extract_samples)

# calculate the area under the curve for each GSR data
df[GSR_list] = df[GSR_list].applymap(str)
ts_ant_gsr, ts_out_gsr = processGSR(df)

# ======================================================================================================================
# Preprocess the GSR data
# ======================================================================================================================

# initialize the data
cleaned_ant_gsr = None
phasic_ant_gsr = None
tonic_ant_gsr = None
cleaned_out_gsr = None
phasic_out_gsr = None
tonic_out_gsr = None

for i, gsr_data in enumerate([ts_ant_gsr, ts_out_gsr]):

    # group the data by participant
    participant_data = {}
    for participant in gsr_data.columns.unique():
        participant_data[participant] = gsr_data[participant]

    # process the GSR data
    preprocessed_gsr = {}
    tonic_gsr = {}
    phasic_gsr = {}

    iteration = 0

    for participant, data in participant_data.items():

        iteration += 1
        sample_rate = 100

        print(f'[{GSR_list[i]}] Processing participant ({iteration}/{len(participant_data)})...')

        # record the number of valid data points in each column
        valid_data = data.count()

        # flatten the data
        data = data.values.flatten()

        # remove NaN values
        data = data[~np.isnan(data)]

        # apply the preprocessing function
        preprocessed = nk.eda_clean(data, sampling_rate=sample_rate)
        sig = nk.eda_phasic(preprocessed, sampling_rate=sample_rate)

        phasic = sig['EDA_Phasic']
        tonic = sig['EDA_Tonic']

        # reverse back to the original shape column-wise using the number of valid data points
        for j, signals in enumerate([preprocessed, phasic, tonic]):
            signals = np.split(signals, np.cumsum(valid_data), axis=0)[:-1]
            signals = [pd.DataFrame(k).reset_index(drop=True) for k in signals]
            signals = pd.concat(signals, axis=1)
            if j == 0:
                preprocessed_gsr[participant] = signals
            elif j == 1:
                phasic_gsr[participant] = signals
            else:
                tonic_gsr[participant] = signals

    if i == 0:
        cleaned_ant_gsr = pd.concat(preprocessed_gsr.values(), axis=1)
        phasic_ant_gsr = pd.concat(phasic_gsr.values(), axis=1)
        tonic_ant_gsr = pd.concat(tonic_gsr.values(), axis=1)

    else:
        cleaned_out_gsr = pd.concat(preprocessed_gsr.values(), axis=1)
        phasic_out_gsr = pd.concat(phasic_gsr.values(), axis=1)
        tonic_out_gsr = pd.concat(tonic_gsr.values(), axis=1)

df['AnticipatoryGSRAUC'] = area_under_curve(cleaned_ant_gsr)
df['OutcomeGSRAUC'] = area_under_curve(cleaned_out_gsr)
df['TonicAnticipatoryGSRAUC'] = area_under_curve(tonic_ant_gsr)
df['PhasicAnticipatoryGSRAUC'] = area_under_curve(phasic_ant_gsr)
df['TonicOutcomeGSRAUC'] = area_under_curve(tonic_out_gsr)
df['PhasicOutcomeGSRAUC'] = area_under_curve(phasic_out_gsr)

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
df.to_csv('./Data/preliminary_data.csv', index=False)


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

