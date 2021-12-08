import math
import h5py
from numpy.lib.stride_tricks import as_strided
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def moving_window_stride(array, window, step):
    """
    Returns view of strided array for moving window calculation with given window size and step
    :param array: numpy.ndarray - input array
    :param window: int - window size
    :param step: int - step lenght
    :return: strided: numpy.ndarray - view of strided array, index: numpy.ndarray - array of indexes
    """
    stride = array.strides[0]
    win_count = math.floor((len(array) - window + step) / step)
    strided = as_strided(array, shape=(win_count, window), strides=(stride*step, stride))
    index = np.arange(window - 1, window + (win_count-1) * step, step)
    return strided, index

def feature_zc(series, window, step, threshold):
    """Zero Crossing"""
    windows_strided, indexes = moving_window_stride(series.values, window, step)
    zc = np.apply_along_axis(lambda x: np.sum(np.diff(x[(x < -threshold) | (x > threshold)] > 0)), axis=1,
                             arr=windows_strided)
    return pd.Series(data=zc, index=series.index[indexes])

def feature_rms(series, window, step):
    """Root Mean Square"""
    windows_strided, indexes = moving_window_stride(series.values, window, step)
    return pd.Series(data=np.sqrt(np.mean(np.square(windows_strided), axis=1)), index=series.index[indexes])

def main():
    signal = pd.read_hdf('train.hdf5')
    ground_thruth = signal[:]['TRAJ_GT']
    ground_thruth.loc[ground_thruth < 0] = np.nan
    nnan = ~np.isnan(ground_thruth)
    signal = signal.loc[nnan]

    temp = feature_rms(signal['EMG_8'], window=500, step=100)
    signal_temp = pd.DataFrame(signal, columns=['TRAJ_GT'], index=temp.index)
    signal_train = pd.DataFrame(signal, columns=['TRAJ_GT'], index=signal_temp.index)

    inv = []
    columns =['EMG_1','EMG_2','EMG_3','EMG_4','EMG_5','EMG_6','EMG_7','EMG_8','EMG_9','EMG_10','EMG_11','EMG_12','EMG_13','EMG_14','EMG_15','EMG_16','EMG_17','EMG_18','EMG_19','EMG_20','EMG_21','EMG_22','EMG_23','EMG_24']
    for i,column in enumerate(columns):
        rms = feature_rms(signal[column], 500, 100)
        zc = feature_zc(signal[column], 500, 100, 0.1)
        signal_train[f'{column}_rms'] = rms
        inv.append(f'{column}_rms')
        signal_train[f'{column}_zc'] = zc
        inv.append(f'{column}_zc')

    valid = pd.read_hdf('valid_blank.hdf5')

    temp = feature_rms(valid['EMG_8'], window=4501, step=500)
    valid_temp = pd.DataFrame(valid, columns=['EMG_1'], index=temp.index)
    valid_prediction = pd.DataFrame(valid, columns=['EMG_1'], index=valid_temp.index)
    valid_prediction = valid_prediction.rename(columns={'EMG_1': 'EMG_1_rms'}, inplace=False)

    for i, column in enumerate(columns):
        rms = feature_rms(valid[column], 4501, 500)
        zc = feature_zc(valid[column], 4501, 500, 0.1)
        valid_prediction[f'{column}_rms'] = rms
        valid_prediction[f'{column}_zc'] = zc
    valid_prediction.drop(valid_prediction.head(1).index, inplace=True)

    clf = LinearDiscriminantAnalysis()
    clf.fit(signal_train[inv], signal_train['TRAJ_GT'])
    predictions = clf.predict(valid_prediction[inv])

    with h5py.File('predictions.hdf5', 'w') as f:
        dset = f.create_dataset("default", data=predictions)

main()