import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from scipy.integrate import trapezoid
import pyphysio as ph
from pyphysio.specialized.eda import DriverEstim, PhasicEstim
import neurokit2 as nk


# extract the GSR data
def extract_samples(d):
    # check if 'GetExperimentSamples' is in the dictionary
    if 'GetExperimentSamples' not in d:
        return d
    else:
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

# ======================================================================================================================
# Define functions to preprocess the GSR data
# ======================================================================================================================
# ------------------------------------------------------------------------------------------------------------------
# The auto method is the default method in NeuroKit2.
# ------------------------------------------------------------------------------------------------------------------
def auto_pipeline(signal, sample_rate, interval=10, amplitude_threshold=0.01, method=None):
    sig, _ = nk.eda_process(signal, sampling_rate=sample_rate, amplitude_min=amplitude_threshold)
    phasic = sig['EDA_Phasic']
    tonic = sig['EDA_Tonic']

    return phasic, tonic


# ------------------------------------------------------------------------------------------------------------------
# NeuroKit2 can implement the three methods for phasic and tonic decomposition: high-pass filter,
# median smoothing,
# sparse deconvolution,
# and convex optimization.
# The change in the method can be done by simply changing the method parameter in the
# eda_phasic function.
# The default method is the cvxeda filter.
# ------------------------------------------------------------------------------------------------------------------
def neurokit_pipeline(signal, sample_rate, interval=10, amplitude_threshold=0.01, method='cvxeda'):
    sig = nk.eda_phasic(signal, sampling_rate=sample_rate, method=method)

    phasic = sig['EDA_Phasic']
    tonic = sig['EDA_Tonic']

    return phasic, tonic


# ------------------------------------------------------------------------------------------------------------------
# The original preprocessing method by Bechara et al., 1999, uses the difference transformation method. There is
# no existing package in Python that implements this method. Therefore, we custom implement it here.
# ------------------------------------------------------------------------------------------------------------------
def difference_pipeline(signal, sample_rate, interval=10, amplitude_threshold=0.01, method=None):
    phasic = pd.DataFrame(difference_transformation(signal, interval=interval, sample_rate=sample_rate))
    # there is no tonic component in the difference transformation method
    tonic = pd.DataFrame(np.zeros(len(phasic)))

    return phasic, tonic


# ------------------------------------------------------------------------------------------------------------------
# However, the continuous deconvolution analysis (CDA) can only be implemented using the pyphysio library.
# ------------------------------------------------------------------------------------------------------------------
def cda_pipeline(signal, sample_rate, interval=10, amplitude_threshold=0.01, method=None):
    # create the data for the driver
    sig = ph.create_signal(signal, sampling_freq=sample_rate)

    driver = DriverEstim()(sig)
    phasic_sig = PhasicEstim(amplitude=amplitude_threshold)(driver)
    tonic_sig = PhasicEstim(amplitude=amplitude_threshold, return_phasic=False)(driver)

    # revert back to the original shape
    phasic = phasic_sig.to_dataframe()['signal_DriverEstim_PhasicEstim']
    tonic = tonic_sig.to_dataframe()['signal_DriverEstim_PhasicEstim']

    return phasic, tonic


# ------------------------------------------------------------------------------------------------------------------
# The combined method applies the most robust method for preprocessing the GSR data. The data first undergoes a
# median smoothing method to remove noise. Then, the data passes a high-pass filter to remove low-frequency drifts.
# Finally, the data undergoes a convex optimization method to separate the phasic and tonic components.
# Could distort the data though.
# ------------------------------------------------------------------------------------------------------------------
def combined_cda_pipeline(signal, sample_rate, interval=10, amplitude_threshold=0.01, method=None):
    # apply the median smoothing method
    sig_smoothed = nk.eda_phasic(signal, sampling_rate=sample_rate,
                                 method='smoothmedian', smoothing_factor=2)['EDA_Phasic']

    # apply the high-pass filter
    sig_highpass = nk.eda_phasic(sig_smoothed, sampling_rate=sample_rate, method='highpass')['EDA_Phasic']

    # apply the cda method
    sig = ph.create_signal(sig_highpass, sampling_freq=sample_rate)

    driver = DriverEstim()(sig)
    phasic_sig = PhasicEstim(amplitude=amplitude_threshold)(driver)
    tonic_sig = PhasicEstim(amplitude=amplitude_threshold, return_phasic=False)(driver)

    # revert back to the original shape
    phasic = phasic_sig.to_dataframe()['signal_DriverEstim_PhasicEstim']
    tonic = tonic_sig.to_dataframe()['signal_DriverEstim_PhasicEstim']

    return phasic, tonic

# define the mapping of method to the corresponding pipeline
method_pipeline = {
    'auto': auto_pipeline,
    'highpass': neurokit_pipeline,
    'smoothmedian': neurokit_pipeline,
    'cvxeda': neurokit_pipeline,
    'SparsEDA': neurokit_pipeline,
    'cda': cda_pipeline,
    'difference': difference_pipeline,
    'combined': combined_cda_pipeline
}
# ======================================================================================================================
# Define functions to preprocess the GSR data based on the smoothing unit
# ======================================================================================================================
def trialwise(ant_gsr, out_gsr, sample_rate, interval, amplitude_threshold, method, max_gsr_samples):
    # combine the anticipatory and outcome GSR data
    combined_gsr = pd.concat([ant_gsr, out_gsr], axis=0)

    # preprocess
    cleaned_gsr = nk.signal_sanitize(nk.eda_clean(combined_gsr, sampling_rate=sample_rate, method='BioSPPy'))

    # apply the cda method
    try:
        phasic_gsr, tonic_gsr = method_pipeline[method](cleaned_gsr, sample_rate, interval, amplitude_threshold,
                                                        method)
    except:
        phasic_gsr = pd.Series(np.nan, index=np.arange(max_gsr_samples))
        tonic_gsr = pd.Series(np.nan, index=np.arange(max_gsr_samples))

    # split the data
    cleaned_ant_gsr = cleaned_gsr[:len(ant_gsr)]
    cleaned_out_gsr = cleaned_gsr[len(ant_gsr):]

    phasic_ant_gsr = phasic_gsr[:len(ant_gsr)]
    phasic_out_gsr = phasic_gsr[len(ant_gsr):]

    tonic_ant_gsr = tonic_gsr[:len(ant_gsr)]
    tonic_out_gsr = tonic_gsr[len(ant_gsr):]

    return cleaned_ant_gsr, cleaned_out_gsr, phasic_ant_gsr, phasic_out_gsr, tonic_ant_gsr, tonic_out_gsr


def trial_split(ant_gsr, out_gsr, sample_rate, interval, amplitude_threshold, method, max_gsr_samples):
    # preprocess
    cleaned_ant_gsr = nk.signal_sanitize(nk.eda_clean(ant_gsr, sampling_rate=100, method='BioSPPy'))
    cleaned_out_gsr = nk.signal_sanitize(nk.eda_clean(out_gsr, sampling_rate=100, method='BioSPPy'))

    # apply the cda method
    try:
        phasic_ant_gsr, tonic_ant_gsr = method_pipeline[method](cleaned_ant_gsr, sample_rate, interval,
                                                                amplitude_threshold, method)
        phasic_out_gsr, tonic_out_gsr = method_pipeline[method](cleaned_out_gsr, sample_rate, interval,
                                                                amplitude_threshold, method)
    except:
        phasic_ant_gsr = pd.Series(np.nan, index=np.arange(max_gsr_samples))
        tonic_ant_gsr = pd.Series(np.nan, index=np.arange(max_gsr_samples))
        phasic_out_gsr = pd.Series(np.nan, index=np.arange(max_gsr_samples))
        tonic_out_gsr = pd.Series(np.nan, index=np.arange(max_gsr_samples))

    return cleaned_ant_gsr, cleaned_out_gsr, phasic_ant_gsr, phasic_out_gsr, tonic_ant_gsr, tonic_out_gsr

unit_mapping = {
    'trial': trialwise,
    'trial_split': trial_split
}
# ======================================================================================================================


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
        # if the trial data is empty, skip the trial
        if trial_data.empty:
            auc_per_second = np.nan
        else:
            trial_data.index = np.arange(10, (len(trial_data) + 1) * 10, 10)
            auc_per_second = trapezoid(trial_data, x=trial_data.index) / (len(trial_data) * 10)
        auc_ant.append(auc_per_second)

    return auc_ant


def find_peak(data):
    peak = []
    for i in range(data.shape[1]):
        trial_data = data.iloc[:, i].dropna()
        if trial_data.empty:
            peak.append(np.nan)
        else:
            peak.append(trial_data.max())
    return peak


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
            data_selected = df[(df['Condition'] == condition) & (df['SetSeen.'] == trial_type)]
            print(ttest_ind(data_selected[data_selected['BestOption'] == 0][GSR_type],
                            data_selected[data_selected['BestOption'] == 1][GSR_type]))
            print('Anticipatory GSR for C:', data_selected[data_selected['BestOption'] == 1][GSR_type].mean())
            print('Anticipatory GSR for A:', data_selected[data_selected['BestOption'] == 0][GSR_type].mean())
