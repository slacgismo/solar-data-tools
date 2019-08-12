# -*- coding: utf-8 -*-
''' Matrix Embedding Module

This module contains functions for embedding PV power time series data into
a matrix

'''

import numpy as np
import pandas as pd

def make_2d(df, key='dc_power', zero_nighttime=True, interp_missing=True,
            trim_start=True, trim_end=True):
    '''
    This function constructs a 2D array (or matrix) from a time series signal with a standardized time axis. The data is
    chunked into days, and each consecutive day becomes a column of the matrix.

    :param df: A pandas data frame contained tabular data with a standardized time axis.
    :param key: The key corresponding to the column in the data frame contained the signal to make into a matrix
    :return: A 2D numpy array with shape (measurements per day, days in data set)
    '''
    if df is not None:
        days = df.resample('D').max().index
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
        if zero_nighttime:
            try:
                with np.errstate(invalid='ignore'):
                    night_msk = D < 0.005 * np.max(D[~np.isnan(D)])
            except ValueError:
                night_msk = D < 0.005 * np.max(D)
            D[night_msk] = np.nan
            good_vals = (~np.isnan(D)).astype(int)
            sunrise_idxs = np.argmax(good_vals, axis=0)
            sunset_idxs = D.shape[0] - np.argmax(np.flip(good_vals, 0), axis=0)
            D_msk = np.zeros_like(D, dtype=np.bool)
            for ix in range(D.shape[1]):
                if sunrise_idxs[ix] > 0:
                    D_msk[:sunrise_idxs[ix] - 1, ix] = True
                    D_msk[sunset_idxs[ix] + 1:, ix] = True
                else:
                    D_msk[:, ix] = True
            D[D_msk] = 0
        if interp_missing:
            D_df = pd.DataFrame(data=D)
            D = D_df.interpolate().values
        return D
    else:
        return