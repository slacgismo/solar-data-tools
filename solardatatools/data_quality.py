# -*- coding: utf-8 -*-
""" Data Quality Checking Module

This module contains functions for identifying corrupt or bad quality data.

"""

import numpy as np
from scipy.stats import mode
from solardatatools.signal_decompositions import tl1_l2d2p365


def make_quality_flags(
    density_scores,
    linearity_scores,
    density_lower_threshold=0.6,
    density_upper_threshold=1.05,
    linearity_threshold=0.1,
):
    density_flags = np.logical_and(
        density_scores > density_lower_threshold,
        density_scores < density_upper_threshold,
    )
    linearity_flags = linearity_scores < linearity_threshold
    return density_flags, linearity_flags


def make_density_scores(
    data_matrix,
    threshold=0.2,
    return_density_signal=False,
    return_fit=False,
    solver=None,
):
    nans = np.isnan(data_matrix)
    capacity_est = np.quantile(data_matrix[~nans], 0.95)
    data_copy = np.copy(data_matrix)
    data_copy[nans] = 0.0
    foo = data_copy > 0.02 * capacity_est
    density_signal = np.sum(foo, axis=0) / data_matrix.shape[0]
    use_days = np.logical_and(density_signal > threshold, density_signal < 0.8)
    fit_signal = tl1_l2d2p365(density_signal, use_ixs=use_days, tau=0.85, solver=solver)
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


def make_linearity_scores(data_matrix, capacity, density_baseline):
    temp_mat = np.copy(data_matrix)
    temp_mat[temp_mat < 0.005 * capacity] = np.nan
    difference_mat = np.round(temp_mat[1:] - temp_mat[:-1], 4)
    modes, counts = mode(difference_mat, axis=0, nan_policy="omit", keepdims=True)
    n = data_matrix.shape[0] - 1
    linearity_scores = np.asarray(counts.data).squeeze() / (n * density_baseline)
    # Label detected infill points with a boolean mask
    infill = np.zeros_like(data_matrix, dtype=bool)
    slct = linearity_scores >= 0.1
    reference_diffs = np.tile(modes[0][slct], (data_matrix.shape[0], 1))
    found_infill = np.logical_or(
        np.isclose(
            np.r_[np.zeros(data_matrix.shape[1]).reshape((1, -1)), difference_mat][
                :, slct
            ],
            reference_diffs,
        ),
        np.isclose(
            np.r_[difference_mat, np.zeros(data_matrix.shape[1]).reshape((1, -1))][
                :, slct
            ],
            reference_diffs,
        ),
    )
    infill[:, slct] = found_infill
    infill_mask = infill
    return linearity_scores, infill_mask


def daily_missing_data_simple(data_matrix, threshold=0.2, return_density_signal=False):
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
    data_copy[nans] = 0.0
    foo = data_copy > 0.005 * capacity_est
    bar = np.sum(foo, axis=0) / data_matrix.shape[0]
    good_days = bar > threshold
    if return_density_signal:
        return good_days, bar
    else:
        return good_days


def dataset_quality_score(
    data_matrix, threshold=0.2, good_days=None, use_advanced=True
):
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
            good_days = make_density_scores(data_matrix, threshold=threshold)
        else:
            good_days = daily_missing_data_simple(data_matrix, threshold=threshold)
    score = np.sum(good_days) / data_matrix.shape[1]
    return score
