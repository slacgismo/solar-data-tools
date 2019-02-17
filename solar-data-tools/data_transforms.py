''' Data Transforms Module

This module contains functions for transforming PV power data, including time-axis standardization and
2D-array generation

'''

from datetime import timedelta
import numpy as np
import pandas as pd

def standardize_time_axis(df, datetimekey='Date-Time'):
    '''
    This function takes in a pandas data frame containing tabular time series data, likely generated with a call to
    pandas.read_csv(). It is assumed that each row of the data frame corresponds to a unique date-time, though not
    necessarily on standard intervals. This function will attempt to convert a user-specified column containing time
    stamps to python datetime objects, assign this column to the index of the data frame, and then standardize the
    index over time. By standardize, we mean reconstruct the index to be at regular intervals, starting at midnight of
    the first day of the data set. This solves a couple common data errors when working with raw data. (1) Missing data
    points from skipped scans in the data acquisition system. (2) Time stamps that are at irregular exact times,
    including fractional seconds.

    :param df: A pandas data frame containing the tabular time series data
    :param datetimekey: An optional key corresponding to the name of the column that contains the time stamps
    :return: A new data frame with a standardized time axis
    '''
    # convert index to timeseries
    try:
        df['Date-Time'] = pd.to_datetime(df['Date-Time'])
        df.set_index('Date-Time', inplace=True)
    except KeyError:
        time_cols = [col for col in df.columns if np.logical_or('Time' in col, 'time' in col)]
        key = time_cols[0]
        df[key] = pd.to_datetime(df[key])
        df.set_index(key, inplace=True)
    # standardize the timeseries axis to a regular frequency over a full set of days
    diff = (df.index[1:] - df.index[:-1]).seconds
    freq = int(np.median(diff))  # the number of seconds between each measurement
    start = df.index[0]
    end = df.index[-1]
    time_index = pd.date_range(start=start.date(), end=end.date() + timedelta(days=1), freq='{}s'.format(freq))[:-1]
    df = df.reindex(index=time_index, method='nearest')
    return df.fillna(value=0)

def make_2d(df, key='dc_power'):
    '''
    This function constructs a 2D array (or matrix) from a time series signal with a standardized time axis. The data is
    chunked into days, and each consecutive day becomes a column of the matrix.

    :param df: A pandas data frame contained tabular data with a standardized time axis.
    :param key: The key corresponding to the column in the data frame contained the signal to make into a matrix
    :return: A 2D numpy array with shape (measurements per day, days in data set)
    '''
    if df is not None:
        n_steps = int(24 * 60 * 60 / df.index.freq.n)
        D = df[key].values.reshape(n_steps, -1, order='F')
        return D
    else:
        return

def fix_time_shifts(data):
    pass