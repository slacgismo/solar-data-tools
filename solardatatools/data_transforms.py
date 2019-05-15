# -*- coding: utf-8 -*-
''' Data Transforms Module

This module contains functions for transforming PV power data, including time-axis standardization and
2D-array generation

'''

from datetime import timedelta
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from sklearn.neighbors.kde import KernelDensity

from solardatatools.clear_day_detection import find_clear_days
from solardatatools.solar_noon import energy_com, avg_sunrise_sunset
from solardatatools.utilities import total_variation_filter,\
    total_variation_plus_seasonal_filter, basic_outlier_filter


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
    try:
        diff = (df.index[1:] - df.index[:-1]).seconds
        freq = int(np.median(diff))  # the number of seconds between each measurement
    except AttributeError:
        diff = df.index[1:] - df.index[:-1]
        freq = np.median(diff) / np.timedelta64(1, 's')
    start = df.index[0]
    end = df.index[-1]
    time_index = pd.date_range(start=start.date(), end=end.date() + timedelta(days=1), freq='{}s'.format(freq))[:-1]
    # This forces the existing data into the closest new timestamp to the
    # old timestamp.
    df = df.reindex(index=time_index, method='nearest', limit=1)
    return df

def make_2d(df, key='dc_power', zero_nighttime=True, interp_missing=True):
    '''
    This function constructs a 2D array (or matrix) from a time series signal with a standardized time axis. The data is
    chunked into days, and each consecutive day becomes a column of the matrix.

    :param df: A pandas data frame contained tabular data with a standardized time axis.
    :param key: The key corresponding to the column in the data frame contained the signal to make into a matrix
    :return: A 2D numpy array with shape (measurements per day, days in data set)
    '''
    if df is not None:
        days = df.resample('D').max().index[1:-1]
        start = days[0]
        end = days[-1]
        try:
            n_steps = int(24 * 60 * 60 / df.index.freq.delta.seconds)
        except AttributeError:
            # No frequency defined for index. Attempt to infer
            freq_ns = np.median(df.index[1:] - df.index[:-1])
            n_steps = int(freq_ns / np.timedelta64(1, 's'))
        D = np.copy(df[key].loc[start:end].iloc[:-1].values.reshape(n_steps, -1, order='F'))
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

def fix_time_shifts(data, verbose=False, return_ixs=False, clear_day_filter=True,
                    c1=10., c2=500., c3=5., solar_noon_estimator='com'):
    '''
    This is an algorithm to detect and fix time stamping shifts in a PV power database. This is a common data error
    that can have a number of causes: improper handling of DST, resetting of a data logger clock, or issues with
    storing the data in the database. The algorithm performs as follows:
    Part 1:
        a) Estimate solar noon for each day relative to the provided time axis. This is estimated as the "center of
           mass" in time of the energy production each day.
        b) Filter this signal for clear days
        c) Fit a total variation filter with seasonal baseline to the output of (b)
        d) Perform KDE-based clustering on the output of (c)
        e) Extract the days on which the transitions between clusters occur
    Part 2:
        a) Find the average solar noon value for each cluster
        b) Taking the first cluster as a reference point, find the offsets in average values between the first cluster
           and all others
        c) Adjust the time axis for all clusters after the first by the amount calculated in (b)

    :param data: A 2D numpy array containing a solar power time series signal (see `data_transforms.make_2d`)
    :param verbose: An option to print information about what clusters are found
    :param return_ixs: An option to return the indices of the boundary days for the clusters
    :return:
    '''
    D = data
    #################################################################################################################
    # Part 1: Detecting the days on which shifts occurs. If no shifts are detected, the algorithm exits, returning
    # the original data array. Otherwise, the algorithm proceeds to Part 2.
    #################################################################################################################
    if solar_noon_estimator == 'com':
        # Find "center of mass" of each day's energy content. This generates a 1D signal from the 2D input signal.
        s1 = energy_com(D)
    elif solar_noon_estimator == 'srsn':
        # estimate solar noon as the average of sunrise time and sunset time
        s1 = avg_sunrise_sunset(D)
    # Apply a clear day filter
    if clear_day_filter:
        m = find_clear_days(D)
        # Filter our obvious outliers in the solar noon / center of mass signal
        msk = basic_outlier_filter(s1[m])
        idxs = np.arange(D.shape[1])
        m[idxs[m][~msk]] = False
        # only keep the entries of the solar noon signal that correspond to clear
        # days that are not outliers
        s1_f = np.empty_like(s1)
        s1_f[:] = np.nan
        s1_f[m] = s1[m]
    else:
        s1_f = s1
    # Apply total variation filter (with seasonal baseline if >1yr of data)
    if len(s1) > 365:
        s2, s_seas = total_variation_plus_seasonal_filter(s1_f, c1=c1, c2=c2)
    else:
        s2 = total_variation_filter(s1_f, C=c3)
    # Perform clustering with KDE
    kde = KernelDensity(kernel='gaussian', bandwidth=0.05).fit(s2[:, np.newaxis])
    X_plot = np.linspace(0.95 * np.min(s2), 1.05 * np.max(s2))[:, np.newaxis]
    log_dens = kde.score_samples(X_plot)
    mins = argrelextrema(log_dens, np.less)[0]      # potential cut points to make clusters
    maxs = argrelextrema(log_dens, np.greater)[0]   # locations of the max point in each cluster
    # Drop clusters with too few members
    keep = np.ones_like(maxs, dtype=np.bool)
    for ix, mx in enumerate(maxs):
        if np.exp(log_dens)[mx] < 1e-1:
            keep[ix] = 0
    mx_keep = maxs[keep]
    mx_drop = maxs[~keep]
    mn_drop = []
    # Determine closest clusters in keep set to each cluster that should be dropped
    for md in mx_drop:
        dists = np.abs(X_plot[:, 0][md] - X_plot[:, 0][mx_keep])
        max_merge = mx_keep[np.argmin(dists)]
        # Determine which minimum index to remove to correctly merge clusters
        for mn in mins:
            cond1 = np.logical_and(mn < max_merge, mn > md)
            cond2 = np.logical_and(mn > max_merge, mn < md)
            if np.logical_or(cond1, cond2):
                if verbose:
                    print('merge', md, 'with', max_merge, 'by dropping', mn)
                mn_drop.append(mn)
    mins_new = np.array([i for i in mins if i not in mn_drop])
    # Assign cluster labels to days in data set
    clusters = np.zeros_like(s1)
    if len(mins_new) > 0:
        for it, ex in enumerate(X_plot[:, 0][mins_new]):
            m = s2 >= ex
            clusters[m] = it + 1
    # Identify indices corresponding to days when time shifts occurred
    index_set = np.arange(D.shape[1]-1)[clusters[1:] != clusters[:-1]] + 1
    # Exit if no time shifts detected
    if len(index_set) == 0:
        if verbose:
            print('No time shifts found')
        if return_ixs:
            return D, []
        else:
            return D
    #################################################################################################################
    # Part 2: Fixing the time shifts.
    #################################################################################################################
    if verbose:
        print('Time shifts found at: ', index_set)
    ixs = np.r_[[None], index_set, [None]]
    # Take averages of solar noon estimates over the segments of the data set defined by the shift points
    A = []
    for i in range(len(ixs) - 1):
        avg = np.average(np.ma.masked_invalid(s1_f[ixs[i]:ixs[i + 1]]))
        A.append(np.round(avg * D.shape[0] / 24))
    A = np.array(A)
    # Considering the first segment as the reference point, determine how much to shift the remaining segments
    rolls = A[0] - A[1:]
    # Apply the corresponding shift to each segment
    Dout = np.copy(D)
    for ind, roll in enumerate(rolls):
        D_rolled = np.roll(D, int(roll), axis=0)
        Dout[:, ixs[ind + 1]:] = D_rolled[:, ixs[ind + 1]:]
    # We find that a second pass with halved weights catches some transition points
    # that might have been missed for data with many small transitions
    Dout = fix_time_shifts(Dout, return_ixs=False, c1=c1/2, c2=c2/2, c3=c3/2)
    if return_ixs:
        return Dout, index_set
    else:
        return Dout
