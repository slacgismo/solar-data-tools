# -*- coding: utf-8 -*-
''' Matrix Embedding Module

This module contains functions for embedding PV power time series data into
a matrix

'''

import numpy as np
import pandas as pd

def find_start_end(arr):
    n = len(arr)
    for i in range(n):
        if arr[i] == False:
            break
    for j in range(-1, -(n + 1), -1):
        if arr[j] == False:
            break
    j += 1
    if j == 0:
        j = None
    return i, j

def make_2d(df, key='dc_power', trim_start=False, trim_end=False,
            return_day_axis=False):
    '''
    This function constructs a 2D array (or matrix) from a time series signal with a standardized time axis. The data is
    chunked into days, and each consecutive day becomes a column of the matrix.

    :param df: A pandas data frame contained tabular data with a standardized time axis.
    :param key: The key corresponding to the column in the data frame contained the signal to make into a matrix
    :return: A 2D numpy array with shape (measurements per day, days in data set)
    '''
    if df is not None:
        days = df.resample('D').first().index
        try:
            n_steps = int(24 * 60 * 60 / df.index.freq.delta.seconds)
        except AttributeError:
            # No frequency defined for index. Attempt to infer
            freq_ns = np.median(df.index[1:] - df.index[:-1])
            freq_delta_seconds = int(freq_ns / np.timedelta64(1, 's'))
            n_steps = int(24 * 60 * 60 / freq_delta_seconds)
        if not trim_start:
            start = days[0].strftime('%Y-%m-%d')
        else:
            start = days[1].strftime('%Y-%m-%d')
        if not trim_end:
            end = days[-1].strftime('%Y-%m-%d')
        else:
            end = days[-2].strftime('%Y-%m-%d')
        D = np.copy(df[key].loc[start:end].values.reshape(n_steps, -1, order='F'))
        # Trim leading or trailing missing days, which can occur when data frame
        # contains data from multiple sources or sensors that have missing
        # values at different times.
        empty_days = np.alltrue(np.isnan(D), axis=0)
        i, j = find_start_end(empty_days)
        D = D[:, i:j]
        day_axis = pd.date_range(start=start, end=end, freq='1D')
        day_axis = day_axis[i:j]
        if return_day_axis:
            return D, day_axis
        else:
            return D
    else:
        return