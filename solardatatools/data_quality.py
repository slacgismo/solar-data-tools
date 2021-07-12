# -*- coding: utf-8 -*-
''' Data Quality Checking Module

This module contains functions for identifying corrupt or bad quality data.

'''

import numpy as np
from solardatatools.signal_decompositions import tl1_l2d2p365

def daily_missing_data_simple(data_matrix, threshold=0.2,
                              return_density_signal=False):
    """
    This function takes a PV power data matrix and returns a boolean array,
    identifying good days. The good days are the ones that are not missing a
    significant amount of data. This assessment is made based on the fraction
    of non-zero and non-NaN values each day. In a typical "good" data set,
    around 40-60% of the measured values each day will be non-zero. The default
    threshold for this function is 20%.

    :param data_matrix: numpy.array, a matrix containing PV power signals
    :param threshold: float, the threshold to identify good days
    :return: a boolean array, with a True if the day passes the test and a
        False if the day fails
    """
    nans = np.isnan(data_matrix)
    capacity_est = np.quantile(data_matrix[~nans], 0.95)
    data_copy = np.copy(data_matrix)
    data_copy[nans] = 0.
    foo = data_copy > 0.005 * capacity_est
    bar = np.sum(foo, axis=0) / data_matrix.shape[0]
    good_days = bar > threshold
    if return_density_signal:
        return good_days, bar
    else:
        return good_days

def daily_missing_data_advanced(data_matrix, threshold=0.2,
                                return_density_signal=False,
                                return_fit=False, solver=None):
    nans = np.isnan(data_matrix)
    capacity_est = np.quantile(data_matrix[~nans], 0.95)
    data_copy = np.copy(data_matrix)
    data_copy[nans] = 0.
    foo = data_copy > 0.02 * capacity_est
    density_signal = np.sum(foo, axis=0) / data_matrix.shape[0]
    use_days = np.logical_and(
        density_signal > threshold, density_signal < 0.8
    )
    fit_signal = tl1_l2d2p365(
        density_signal,
        use_ixs=use_days,
        tau=0.85,
        solver=solver
    )
    scores = density_signal / fit_signal
    out = [scores]
    if return_density_signal:
        out.append(density_signal)
    if return_fit:
        out.append(fit_signal)
    if len(out) == 1:
        out = out[0]
    else:
        out = tuple(out)
    return out

def dataset_quality_score(data_matrix, threshold=0.2, good_days=None,
                          use_advanced=True):
    """
    This function scores a complete data set. The score is the fraction of days
    in the data set that pass the missing data test. A score of 1 means all the
    days in the data set pass the test and are not missing data.

    :param data_matrix: numpy.array, a matrix containing PV power signals
    :param threshold: float, the threshold to identify good days
    :return: the score, a float between 0 and 1
    """
    if good_days is None:
        if use_advanced:
            good_days = daily_missing_data_advanced(data_matrix, threshold=threshold)
        else:
            good_days = daily_missing_data_simple(data_matrix, threshold=threshold)
    score = np.sum(good_days) / data_matrix.shape[1]
    return score
