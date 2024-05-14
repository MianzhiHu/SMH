import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import requests
import seaborn as sns
import json
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

# apply the preprocessing function to each column of the dataframes
tonic_ant_gsr = []
tonic_out_gsr = []
phasic_ant_gsr = []
phasic_out_gsr = []

for i, gsr_data in enumerate([ts_ant_gsr, ts_out_gsr]):
    for j in range(gsr_data.shape[1]):

        batch_size = 500

        if j % batch_size == 0:
            print(f'Processing column {j}/{gsr_data.shape[1]} for {GSR_list[i]}...')

        sample_rate = 100
        new_sample_rate = 100
        segment_width = (gsr_data.iloc[:, j].notnull().sum() - 1) / 100

        # remove Nan values
        clean_data = gsr_data.iloc[:, j].dropna()


        # Redirect stdout to suppress unwanted prints
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            wd = process_statistical(clean_data, use_scipy=True, sample_rate=sample_rate,
                                     new_sample_rate=new_sample_rate,
                                     segment_width=segment_width, segment_overlap=0)[1]


        tonic_gsr = wd['tonic_gsr'][0]
        phasic_gsr = wd['filtered_phasic_gsr'][0]

        # save the data
        if i == 0:
            tonic_ant_gsr.append(tonic_gsr)
            phasic_ant_gsr.append(phasic_gsr)
        else:
            tonic_out_gsr.append(tonic_gsr)
            phasic_out_gsr.append(phasic_gsr)


# calculate the area under the curve for each GSR data
gsr_list = [tonic_ant_gsr, phasic_ant_gsr, tonic_out_gsr, phasic_out_gsr]

# convert all to dataframes
for i, gsr_data in enumerate(gsr_list):
    gsr_list[i] = pd.DataFrame(gsr_data).T

# unnest the dataframes from the list
tonic_ant_gsr, phasic_ant_gsr, tonic_out_gsr, phasic_out_gsr = gsr_list


df['AnticipatoryGSRAUC'] = area_under_curve(phasic_ant_gsr)
df['OutcomeGSRAUC'] = area_under_curve(phasic_out_gsr)

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

# # save the data
df.to_csv('./Data/preliminary_data.csv', index=False)


# ======================================================================================================================
# for illustration, we will use the average of the anticipatory GSR data to show the preprocessing steps
# ======================================================================================================================
# preprocess the data
m, wd, eda_clean = process_statistical(ts_ant_gsr.mean(axis=1), use_scipy=True, sample_rate=100, new_sample_rate=100,
                                       segment_width=5, segment_overlap=0)

original = ts_ant_gsr.mean(axis=1)
tonic_gsr = pd.DataFrame(wd['tonic_gsr'][0])
phasic_gsr = pd.DataFrame(wd['filtered_phasic_gsr'][0])
# multiply the row index by 10 to get the time in ms
tonic_gsr.index = tonic_gsr.index * 10
phasic_gsr.index = phasic_gsr.index * 10


# Plot the original and filtered signals
palette = sns.color_palette("husl", 3)
sns.set_style("white")
plt.figure()
plt.plot(original, color=palette[0], label='Original Signal')
plt.tight_layout()
sns.despine()
plt.savefig('./figures/preprocessing_original.png', dpi=300)
plt.show()

plt.figure()
plt.plot(tonic_gsr, color=palette[1], label='Tonic GSR')
plt.tight_layout()
sns.despine()
plt.savefig('./figures/preprocessing_tonic.png', dpi=300)
plt.show()

plt.figure()
plt.plot(phasic_gsr, color=palette[2], label='Phasic GSR')
plt.tight_layout()
sns.despine()
plt.savefig('./figures/preprocessing_phasic.png', dpi=300)
plt.show()


