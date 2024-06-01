import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from scipy.integrate import trapz


# extract the GSR data
def extract_samples(d):
    return d['GetExperimentSamples'][0][2:]  # Skip the first two elements ('GSR' and 1)


def processGSR(df, standardize=None, draw=False, separate=False):
    def processIndividualGSR(data, column, standardize=None):
        ts_gsr = data[column].apply(
            lambda x: np.array(x.replace('[', '').replace(']', '').split(','), dtype=float))
        ts_gsr = pd.DataFrame(ts_gsr.tolist(), index=ts_gsr.index).T
        ts_gsr.index = ts_gsr.index * 10
        if standardize == 'subject':
            for i in range(ts_gsr.shape[1]):
                ts_gsr.iloc[:, i] = (ts_gsr.iloc[:, i] - ts_gsr.iloc[:, i].mean()) / ts_gsr.iloc[:, i].std()
        elif standardize == 'all':
            ts_gsr = (ts_gsr - ts_gsr.mean().mean()) / ts_gsr.std().std()
        else:
            pass

        return ts_gsr

    ts_ant_gsr = processIndividualGSR(df, 'AnticipatoryGSR', standardize)
    ts_out_gsr = processIndividualGSR(df, 'OutcomeGSR', standardize)

    if draw:
        if separate:
            plt.figure()
            plt.plot(ts_ant_gsr.mean(axis=1))
            plt.xlabel('Time (ms)')
            plt.ylabel('Anticipatory GSR')
            plt.show()

            plt.figure()
            plt.plot(ts_out_gsr.mean(axis=1))
            plt.xlabel('Time (ms)')
            plt.ylabel('Outcome GSR')
            plt.show()

        else:
            plt.figure()
            plt.plot(ts_ant_gsr.mean(axis=1), label='Anticipatory GSR')
            plt.plot(ts_out_gsr.mean(axis=1), label='Outcome GSR')
            plt.xlabel('Time (ms)')
            plt.ylabel('GSR')
            plt.legend()
            plt.show()

    return ts_ant_gsr, ts_out_gsr


# Rename the columns to include the participant number and trial number
def rename_columns(df):
    new_columns = []
    for participant in df.columns.unique():
        trial_number = 1
        for _ in df.loc[:, participant].columns:
            new_columns.append(f"{participant}/trial{trial_number}")
            trial_number += 1
    df.columns = new_columns
    return df


# Function to interleave the columns of anticipatory and outcome GSR signals to perform preprocessing
def interleave_columns(df1, df2):
    interleaved = []
    for col1, col2 in zip(df1.columns, df2.columns):
        interleaved.append(df1[col1])
        interleaved.append(df2[col2])
    return pd.concat(interleaved, axis=1)


# Function to calculate the difference (same as implemented by Bechara et al., 1999)
def difference_transformation(data, interval, sample_rate):
    # Calculate the time interval
    time_interval = interval / sample_rate
    # Create an array to store the transformed data
    transformed_data = np.zeros(len(data) - interval)

    for i in range(len(data) - interval):
        # Calculate the difference in amplitude
        amplitude_difference = data[i + interval] - data[i]
        # Calculate the difference per time interval
        transformed_data[i] = amplitude_difference / time_interval

    return transformed_data


# Function to unnest the combined gsr signals back into separate anticipatory and outcome gsr signals
def unzip_combined_data(df):
    # take the first column as the ant_gsr
    anticipatory_gsr = df.iloc[:, ::2]

    # take the second column as the out_gsr
    outcome_gsr = df.iloc[:, 1::2]

    return anticipatory_gsr, outcome_gsr


def area_under_curve(data):
    auc_ant = []

    for i in range(data.shape[1]):
        trial_data = data.iloc[:, i].dropna()
        trial_data.index = np.arange(10, (len(trial_data) + 1) * 10, 10)
        auc_per_second = trapz(trial_data, x=trial_data.index) / (len(trial_data) * 10)
        auc_ant.append(auc_per_second)

    return auc_ant


# Function to apply the mapping
def check_best_option(row, mapping):
    trial = row['SetSeen ']
    response = row['KeyResponse']
    if trial in mapping and response == mapping[trial]:
        return 1  # Best option chosen
    return 0  # Best option not chosen


def pairwise_t_test_GSR(df, GSR_type, data_type, trial_type=None):
    for condition in ['Baseline', 'Frequency', 'Magnitude']:
        print(condition)

        if data_type == 'training':
            data_selected = df[df['Condition'] == condition]
            for phase in ['Phase 1', 'Phase 2', 'Phase 3', 'Phase 4', 'Phase 5', 'Phase 6']:
                print(phase)
                print(ttest_ind(
                    data_selected[(data_selected['BestOption'] == 0) & (data_selected['Phase'] == phase)][GSR_type],
                    data_selected[(data_selected['BestOption'] == 1) & (data_selected['Phase'] == phase)][GSR_type]))
                print('Outcome GSR for the optimal option:',
                      data_selected[(data_selected['BestOption'] == 1) & (data_selected['Phase'] == phase)][
                          'OutcomeGSRAUC'].mean())
                print('Outcome GSR for the suboptimal option:',
                      data_selected[(data_selected['BestOption'] == 0) & (data_selected['Phase'] == phase)][
                          'OutcomeGSRAUC'].mean())
                print()

        elif data_type == 'testing':
            data_selected = df[(df['Condition'] == condition) & (df['SetSeen '] == trial_type)]
            print(ttest_ind(data_selected[data_selected['BestOption'] == 0][GSR_type],
                            data_selected[data_selected['BestOption'] == 1][GSR_type]))
            print('Anticipatory GSR for C:', data_selected[data_selected['BestOption'] == 1][GSR_type].mean())
            print('Anticipatory GSR for A:', data_selected[data_selected['BestOption'] == 0][GSR_type].mean())
