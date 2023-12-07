""" Utilities Module

This module contains utility functions used by other modules.

"""

import sys
import numpy as np
import cvxpy as cvx
from scipy.interpolate import interp1d


def basic_outlier_filter(x, outlier_constant=1.5):
    """
    Applies an outlier filter based on the interquartile range definition:
        any data point more than 1.5 interquartile ranges (IQRs) below the
        first quartile or above the third quartile

    Function returns a boolean mask for entries in the input array that are
    not outliers.

    :param x: ndarray
    :param outlier_constant: float, multiplier constant on IQR
    :return: boolean mask
    """
    a = np.array(x)
    upper_quartile = np.percentile(a, 75)
    lower_quartile = np.percentile(a, 25)
    iqr = (upper_quartile - lower_quartile) * outlier_constant
    quartile_set = (lower_quartile - iqr, upper_quartile + iqr)
    mask = np.logical_and(a >= quartile_set[0], a <= quartile_set[1])
    return mask


def progress(count, total, status="", bar_length=60):
    """
    Python command line progress bar in less than 10 lines of code. · GitHub
    https://gist.github.com/vladignatyev/06860ec2040cb497f0f3
    :param count: the current count, int
    :param total: to total count, int
    :param status: a message to display
    :return:
    """
    bar_len = bar_length
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = "=" * filled_len + "-" * (bar_len - filled_len)

    sys.stdout.write("[%s] %s%s ...%s\r" % (bar, percents, "%", status))
    sys.stdout.flush()


def find_runs(x):
    """Find runs of consecutive items in an array.
    https://gist.github.com/alimanfoo/c5977e87111abe8127453b21204c1065"""

    # ensure array
    x = np.asanyarray(x)
    if x.ndim != 1:
        raise ValueError("only 1D array supported")
    n = x.shape[0]

    # handle empty array
    if n == 0:
        return np.array([]), np.array([]), np.array([])

    else:
        # find run starts
        loc_run_start = np.empty(n, dtype=bool)
        loc_run_start[0] = True
        np.not_equal(x[:-1], x[1:], out=loc_run_start[1:])
        run_starts = np.nonzero(loc_run_start)[0]

        # find run values
        run_values = x[loc_run_start]

        # find run lengths
        run_lengths = np.diff(np.append(run_starts, n))

        return run_values, run_starts, run_lengths


def time_dilate(data, mask, power=8, scale=None):
    """Process an entire PV power matrix at once
    :return:
    """
    if scale is None:
        scale = 1
    N = 2**power
    output = np.zeros((N, data.shape[1]))
    xs_new = np.linspace(0, 1, N)
    for col_ix in range(data.shape[1]):
        y = data[:, col_ix] * scale
        msk = mask[:, col_ix]
        xs = np.linspace(0, 1, int(np.sum(msk)))
        interp_f = interp1d(xs, y[msk])
        resampled_signal = interp_f(xs_new)
        if np.sum(resampled_signal) > 0:
            output[:, col_ix] = resampled_signal
        else:
            output[:, col_ix] = 0
    return output


def undo_time_dilate(data, mask, scale=None):
    if scale is None:
        scale = 1
    output = np.zeros_like(mask, dtype=float)
    xs_old = np.linspace(0, 1, data.shape[0])
    for col_ix in range(data.shape[1]):
        # the number of non-zero data points on that day
        n_pts = np.sum(mask[:, col_ix])
        # a linear space of n_pts values between 0 and 1
        xs_new = np.linspace(0, 1, n_pts)
        # the mapping from the old index to the data
        interp_f = interp1d(xs_old, data[:, col_ix] / scale)
        # transform the signal to the new length
        resampled_signal = interp_f(xs_new)
        # insert the data into the daytime index values
        output[mask[:, col_ix], col_ix] = resampled_signal
    return output
