import pandas as pd
import ast
import numpy as np
import json
from utilities.utility_processGSR import processGSR, area_under_curve

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
def extract_samples(d):
    return d['GetExperimentSamples'][0][2:]  # Skip the first two elements ('GSR' and 1)


df[GSR_list] = df[GSR_list].applymap(extract_samples)


# calculate the area under the curve for each GSR data
df[GSR_list] = df[GSR_list].applymap(str)
ts_ant_gsr, ts_out_gsr = processGSR(df)
df['AnticipatoryGSRAUC'] = area_under_curve(ts_ant_gsr)
df['OutcomeGSRAUC'] = area_under_curve(ts_out_gsr)

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
# Function to apply the mapping
def check_best_option(row, mapping):
    trial = row['SetSeen ']
    response = row['KeyResponse']
    if trial in mapping and response == mapping[trial]:
        return 1  # Best option chosen
    return 0  # Best option not chosen


# Apply the function using DataFrame.apply()
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
