import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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


def area_under_curve(data):

    auc_ant = []

    for i in range(data.shape[1]):
        trial_data = data.iloc[:, i].dropna()
        trial_data.index = np.arange(10, (len(trial_data) + 1) * 10, 10)
        auc_per_second = np.trapz(trial_data, x=trial_data.index) / (len(trial_data) * 10)
        auc_ant.append(auc_per_second)

    return auc_ant