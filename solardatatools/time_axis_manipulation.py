# -*- coding: utf-8 -*-
''' Data Transforms Module

This module contains functions for transforming PV power data, including time-axis standardization and
2D-array generation

'''

from datetime import timedelta
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from scipy.stats import mode
from sklearn.neighbors import KernelDensity

from solardatatools.clear_day_detection import find_clear_days
from solardatatools.solar_noon import energy_com, avg_sunrise_sunset
from solardatatools.utilities import total_variation_filter,\
    total_variation_plus_seasonal_filter, basic_outlier_filter

TZ_LOOKUP = {
    'America/Anchorage': 9,
    'America/Chicago': 6,
    'America/Denver': 7,
    'America/Los_Angeles': 8,
    'America/New_York': 5,
    'America/Phoenix': 7,
    'Pacific/Honolulu': 10,
    'Canada/Central': 6
}


def make_time_series(df, return_keys=True, localize_time=-8, timestamp_key='ts',
                     value_key='meas_val_f', name_key='meas_name',
                     groupby_keys=['site', 'sensor'],
                     filter_length=200):
    '''
    Accepts a Pandas data frame extracted from a relational or Cassandra database.
    These queries often result in data with repeated timestamps, as you might
    have multiple columns stacked into rows in the database. Defaults are
    intended to work with GISMo's VADER Cassandra database implementation.

    Returns a data frame with a single timestamp
    index and the data from different systems split into columns.

    :param df: A Pandas data from generated from a query the VADER Cassandra database
    :param return_keys: If true, return the mapping from data column names to site and system ID
    :param localize_time: If non-zero, localize the time stamps. Default is PST or UTC-8
    :param filter_length: The number of non-null data values a single system must have to be included in the output
    :return: A time-series data frame
    '''
    # Make sure that the timestamps are monotonically increasing. There may be
    # missing or repeated time stamps
    df.sort_values(timestamp_key, inplace=True)
    # Determine the start and end times
    start = df.iloc[0][timestamp_key]
    end = df.iloc[-1][timestamp_key]
    time_index = pd.to_datetime(df['ts'].sort_values())
    time_index = time_index[~time_index.duplicated(keep='first')]
    output = pd.DataFrame(index=time_index)
    site_keys = []
    site_keys_a = site_keys.append
    grouped = df.groupby(groupby_keys)
    keys = grouped.groups.keys()
    counter = 1
    for key in keys:
        df_view = df.loc[grouped.groups[key]]
        ############## data cleaning ####################################
        #df_view = df_view[pd.notnull(df_view[value_key])]               # Drop records with nulls
        df_view.set_index(timestamp_key, inplace=True)                  # Make the timestamp column the index
        df_view.index = pd.to_datetime(df_view.index)
        df_view.sort_index(inplace=True)                                # Sort on time
        df_view = df_view[~df_view.index.duplicated(keep='first')]      # Drop duplicate times
        df_view.reindex(index=time_index, method=None)                  # Match the master index, interp missing
        #################################################################
        meas_name = str(df_view[name_key][0])
        col_name = meas_name + '_{:02}'.format(counter)
        output[col_name] = df_view[value_key]
        if output[col_name].count() > filter_length:  # final filter on low data count relative to time index
            site_keys_a((key, col_name))
            counter += 1
        else:
            del output[col_name]
    if localize_time:
        output.index = output.index + pd.Timedelta(hours=localize_time)  # Localize time

    if return_keys:
        return output, site_keys
    else:
        return output

def standardize_time_axis(df, datetimekey='Date-Time', timeindex=True):
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
    if not timeindex:
        try:
            df[datetimekey] = pd.to_datetime(df[datetimekey])
            df.set_index(datetimekey, inplace=True)
        except KeyError:
            time_cols = [col for col in df.columns if np.logical_or('Time' in col, 'time' in col)]
            key = time_cols[0]
            df[datetimekey] = pd.to_datetime(df[key])
            df.set_index(datetimekey, inplace=True)
    # standardize the timeseries axis to a regular frequency over a full set of days
    df.index = pd.to_datetime(df.index)
    try:
        diff = (df.index[1:] - df.index[:-1]).seconds
        freq = int(np.median(diff[~np.isnan(diff)]))  # the number of seconds between each measurement
    except AttributeError:
        diff = df.index[1:] - df.index[:-1]
        freq = np.median(diff) / np.timedelta64(1, 's')
    start = df.index[0]
    end = df.index[-1]

    time_index = pd.date_range(start=start.date(), end=end.date() + timedelta(days=1), freq='{}s'.format(freq))[:-1]
    # This forces the existing data into the closest new timestamp to the
    # old timestamp.
    df = df.loc[df.index.notnull()]\
            .reindex(index=time_index, method='nearest', limit=1)
    return df

def fix_daylight_savings_with_known_tz(df, tz='America/Los_Angeles', inplace=False):
    index = df.index.tz_localize(tz, nonexistent='NaT', ambiguous='NaT')\
                .tz_convert('Etc/GMT+{}'.format(TZ_LOOKUP[tz]))\
                .tz_localize(None)
    if inplace:
        df.index = index
        return
    else:
        df_out = df.copy()
        df_out.index = index
        return df_out
