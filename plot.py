from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from utils import pkl_load

def plot_raw(data_name, target, include_sep=False):
    df = pd.read_csv(f'datasets/{data_name}.csv', index_col='date', parse_dates=True)
    df = df[[target]]
    plt.plot(df)
    if include_sep:
        train = int(0.6 * len(df))
        val = int(0.8 * len(df))
        plt.vlines(x=[df.index[train], df.index[val]], ymin=min(df[target]), ymax=max(df[target]), colors=["blue", "red"])
    plt.show()

def plot_pred(filename, data_name, target, pred_len, x_label='', y_label='', seed=None):
    # load files
    pkl = pkl_load(filename)
    df = pd.read_csv(f'datasets/{data_name}.csv', index_col='date', parse_dates=True)
    df=df[[target]]
    val = int(0.8 * len(df))

    pred = pkl[pred_len]['raw']
    pred = pred.reshape(pred.shape[1], pred.shape[2])           # (1, n, pred_len, 1) -> (1, pred_len)
    tru = pkl[pred_len]['raw_gt']
    tru = tru.reshape(tru.shape[1], tru.shape[2])

    # randomly sample 16 test points
    if seed: np.random.seed(seed)
    smpl = np.random.randint(0, pred.shape[0], 16)
    smpl.sort()

    fig, axs = plt.subplots(4, 4, sharex=True)
    for i in range(16):
        axs[i // 4, i % 4].plot(pred[smpl[i]])
        axs[i // 4, i % 4].plot(tru[smpl[i]])
        axs[i // 4, i % 4].set_title(f'{df.index[val+1+smpl[i]]}')
    fig.legend(['pred', 'gt'])
    fig.text(0.5, 0.04, x_label, ha='center', va='center')
    fig.text(0.06, 0.5, y_label, ha='center', va='center', rotation='vertical')
    fig.set_size_inches(12, 10)

    fig.suptitle(f'Length-{pred_len} CoST Predictions for {data_name} Dataset')

    # plt.plot(pred[1000,:], label="pred")
    # plt.plot(tru[1000,:], label="gt")
    # plt.legend()
    plt.show()

def plot_both(fcsv, fpkl, target, pred_len):
    # plot true data
    df = pd.read_csv(fcsv, index_col="date", parse_dates=True)
    df = df[[target]]
    plt.plot(df)
    train = int(0.6 * len(df))
    val = int(0.8 * len(df))
    plt.vlines(x=[df.index[train], df.index[val]], ymin=min(df[target]), ymax=max(df[target]), colors=["blue", "red"])

    # plot pred
    pkl = pkl_load(fpkl)
    predictions = pkl[pred_len]['raw']
    predictions = predictions.reshape(predictions.shape[1], predictions.shape[2])           # (1, n, pred_len, 1) -> (n, pred_len)
    plt.plot([df.index[val+i] for i in range(pred_len)], predictions[0])
    plt.show()

if __name__ == "__main__":

    # WTH dataset
    fpkl = "training/WTH/forecast_univar_20240603_100510/out.pkl"
    data_name = 'WTH'
    target = 'WetBulbCelsius'

    # plot_pred(fpkl, data_name, target, 24, x_label='Hourly Time', y_label='Wet Bulb (Celsius)', seed=207)
    # plot_pred(fpkl, data_name, target, 168, x_label='Hourly Time', y_label='Wet Bulb (Celsius)', seed=207)
    # plot_pred(fpkl, data_name, target, 720, x_label='Hourly Time', y_label='Wet Bulb (Celsius)', seed=207)

    # ETTh1 dataset
    fpkl = 'training/ETTh2/forecast_univar_20240603_102610/out.pkl'
    data_name = 'ETTh2'
    target = 'OT'

    # plot_raw(data_name, target, True)

    plot_pred(fpkl, data_name, target, 24, x_label='Hourly Time', y_label='Oil Temperature (Celsius)', seed=207)
    plot_pred(fpkl, data_name, target, 168, x_label='Hourly Time', y_label='Oil Temperature (Celsius)', seed=207)
    plot_pred(fpkl, data_name, target, 720, x_label='Hourly Time', y_label='Oil Temperature (Celsius)', seed=207)