# -*- coding: utf-8 -*-
''' Data Quality Checking Module

This module contains functions for identifying corrupt or bad quality data.

'''

import numpy as np

def daily_missing_data(data_matrix, threshold=0.2):
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
    return bar > threshold


def dataset_quality_score(data_matrix, threshold=0.2):
    """
    This function scores a complete data set. The score is the fraction of days
    in the data set that pass the missing data test. A score of 1 means all the
    days in the data set pass the test and are not missing data.

    :param data_matrix: numpy.array, a matrix containing PV power signals
    :param threshold: float, the threshold to identify good days
    :return: the score, a float between 0 and 1
    """
    good_days = daily_missing_data(data_matrix, threshold=threshold)
    score = np.sum(good_days) / data_matrix.shape[1]
    return score
