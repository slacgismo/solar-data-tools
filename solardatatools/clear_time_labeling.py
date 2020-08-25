# -*- coding: utf-8 -*-
''' Clear Time Labeling Module

This module contains a function to find time periods in a data matrix
that correspond to clear sky output..

'''
import numpy as np
from solardatatools.utilities import find_runs

def find_clear_times(measured_matrix, clear_matrix, capacity_estimate,
                     th_relative_power=0.1, th_relative_smoothness=0.05,
                     min_length=3):
    n1, n2 = measured_matrix.shape
    # calculate clearness index based on clear sky power estimates
    ci = np.zeros_like(clear_matrix)
    daytime = np.logical_and(
        measured_matrix >= 0.05 * np.percentile(clear_matrix, 95),
        clear_matrix >= 0.05 * np.percentile(clear_matrix, 95)
    )
    ci[daytime] = np.clip(np.divide(measured_matrix[daytime], clear_matrix[daytime]), 0, 2)
    # compare relative 2nd order smoothness of measured data and clear sky estimate
    diff_meas = np.r_[
        0,
        (np.diff(measured_matrix.ravel(order='F') / capacity_estimate, n=2)),
        0
    ]
    diff_clear = np.r_[
        0,
        (np.diff(clear_matrix.ravel(order='F') / capacity_estimate, n=2)),
        0
    ]
    diff_compare = (np.abs(diff_meas - diff_clear)).reshape(n1, n2, order='F')
    # assign clear times as high clearness index and similar smoothness
    clear_times = np.logical_and(
        np.abs(ci - 1) <= th_relative_power,
        diff_compare <= th_relative_smoothness
    )
    # remove clear times that are in small groups, below the minimum length threshold
    run_values, run_starts, run_lengths = find_runs(clear_times.ravel(order='F'))
    for val, start, length in zip(run_values, run_starts, run_lengths):
        if val is False:
            continue
        if length >= min_length:
            continue
        i = start % n1
        j = start // n1
        for count in range(length):
            clear_times[i + count, j] = False
    return clear_times