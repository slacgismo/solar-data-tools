# -*- coding: utf-8 -*-
''' Utilities Module

This module contains utility function used by other modules.

'''
import sys
import numpy as np
import cvxpy as cvx

def total_variation_filter(signal, C=5):
    '''
    This function performs total variation filtering or denoising on a 1D signal. This filter is implemented as a
    convex optimization problem which is solved with cvxpy.
    (https://en.wikipedia.org/wiki/Total_variation_denoising)

    :param signal: A 1d numpy array (must support boolean indexing) containing the signal of interest
    :param C: The regularization parameter to control the total variation in the final output signal
    :return: A 1d numpy array containing the filtered signal
    '''
    s_hat = cvx.Variable(len(signal))
    mu = cvx.Constant(value=C)
    index_set = ~np.isnan(signal)
    objective = cvx.Minimize(cvx.sum(cvx.huber(signal[index_set] - s_hat[index_set]))
                             + mu * cvx.norm1(cvx.diff(s_hat, k=1)))
    problem = cvx.Problem(objective=objective)
    try:
        problem.solve(solver='MOSEK')
    except Exception as e:
        print(e)
        print('Trying ECOS solver')
        problem.solve(solver='ECOS')
    return s_hat.value

def total_variation_plus_seasonal_filter(signal, c1=10, c2=500,
                                         residual_weights=None, tv_weights=None,
                                         use_ixs=None):
    '''
    This performs total variation filtering with the addition of a seasonal baseline fit. This introduces a new
    signal to the model that is smooth and periodic on a yearly time frame. This does a better job of describing real,
    multi-year solar PV power data sets, and therefore does an improved job of estimating the discretely changing
    signal.

    :param signal: A 1d numpy array (must support boolean indexing) containing the signal of interest
    :param c1: The regularization parameter to control the total variation in the final output signal
    :param c2: The regularization parameter to control the smoothness of the seasonal signal
    :return: A 1d numpy array containing the filtered signal
    '''
    if residual_weights is None:
        residual_weights = np.ones_like(signal)
    if tv_weights is None:
        tv_weights = np.ones(len(signal) - 1)
    if use_ixs is None:
        index_set = ~np.isnan(signal)
    else:
        index_set = use_ixs
    s_hat = cvx.Variable(len(signal))
    s_seas = cvx.Variable(len(signal))
    s_error = cvx.Variable(len(signal))
    c1 = cvx.Constant(value=c1)
    c2 = cvx.Constant(value=c2)
    #w = len(signal) / np.sum(index_set)
    objective = cvx.Minimize(
        # (365 * 3 / len(signal)) * w *
        # cvx.sum(cvx.huber(cvx.multiply(residual_weights, s_error)))
        10 * cvx.norm(cvx.multiply(residual_weights, s_error))
        + c1 * cvx.norm1(cvx.multiply(tv_weights, cvx.diff(s_hat, k=1)))
        + c2 * cvx.norm(cvx.diff(s_seas, k=2))
        # + c2 * .1 * cvx.norm(cvx.diff(s_seas, k=1))
    )
    constraints = [
        signal[index_set] == s_hat[index_set] + s_seas[index_set] + s_error[index_set],
        cvx.sum(s_seas[:365]) == 0
    ]
    if len(signal) > 365:
        constraints.append(s_seas[365:] - s_seas[:-365] == 0)
    problem = cvx.Problem(objective=objective, constraints=constraints)
    problem.solve()
    return s_hat.value, s_seas.value

def local_median_regression_with_seasonal(signal, use_ixs=None, c1=1e3, solver='ECOS'):
    '''
    for a list of available solvers, see:
        https://www.cvxpy.org/tutorial/advanced/index.html#solve-method-options

    :param signal: 1d numpy array
    :param use_ixs: optional index set to apply cost function to
    :param c1: float
    :param solver: string
    :return: median fit with seasonal baseline removed
    '''
    if use_ixs is None:
        use_ixs = np.arange(len(signal))
    x = cvx.Variable(len(signal))
    objective = cvx.Minimize(
        cvx.norm1(signal[use_ixs] - x[use_ixs]) + c1 * cvx.norm(cvx.diff(x, k=2))
    )
    if len(signal) > 365:
        constraints = [
            x[365:] == x[:-365]
        ]
    else:
        constraints = []
    prob = cvx.Problem(objective, constraints=constraints)
    prob.solve(solver=solver)
    return x.value

def local_quantile_regression_with_seasonal(signal, use_ixs=None, tau=0.75,
                                            c1=1e3, solver='ECOS',
                                            residual_weights=None,
                                            tv_weights=None):
    '''
    https://colab.research.google.com/github/cvxgrp/cvx_short_course/blob/master/applications/quantile_regression.ipynb

    :param signal: 1d numpy array
    :param use_ixs: optional index set to apply cost function to
    :param tau: float, parameter for quantile regression
    :param c1: float
    :param solver: string
    :return: median fit with seasonal baseline removed
    '''
    if use_ixs is None:
        use_ixs = np.arange(len(signal))
    x = cvx.Variable(len(signal))
    r = signal[use_ixs] - x[use_ixs]
    objective = cvx.Minimize(
        cvx.sum(0.5 * cvx.abs(r) + (tau - 0.5) * r) + c1 * cvx.norm(cvx.diff(x, k=2))
    )
    if len(signal) > 365:
        constraints = [
            x[365:] == x[:-365]
        ]
    else:
        constraints = []
    prob = cvx.Problem(objective, constraints=constraints)
    prob.solve(solver=solver)
    return x.value


def total_variation_plus_seasonal_quantile_filter(signal, use_ixs=None, tau=0.995,
                                                  c1=1e3, c2=1e2, c3=1e2,
                                                  solver='ECOS',
                                                  residual_weights=None,
                                                  tv_weights=None):
    '''
    This performs total variation filtering with the addition of a seasonal baseline fit. This introduces a new
    signal to the model that is smooth and periodic on a yearly time frame. This does a better job of describing real,
    multi-year solar PV power data sets, and therefore does an improved job of estimating the discretely changing
    signal.

    :param signal: A 1d numpy array (must support boolean indexing) containing the signal of interest
    :param c1: The regularization parameter to control the total variation in the final output signal
    :param c2: The regularization parameter to control the smoothness of the seasonal signal
    :return: A 1d numpy array containing the filtered signal
    '''
    n = len(signal)
    if residual_weights is None:
        residual_weights = np.ones_like(signal)
    if tv_weights is None:
        tv_weights = np.ones(len(signal) - 1)
    if use_ixs is None:
        use_ixs = np.ones(n, dtype=np.bool)
    # selected_days = np.arange(n)[index_set]
    # np.random.shuffle(selected_days)
    # ix = 2 * n // 3
    # train = selected_days[:ix]
    # validate = selected_days[ix:]
    # train.sort()
    # validate.sort()

    s_hat = cvx.Variable(n)
    s_seas = cvx.Variable(n)
    s_error = cvx.Variable(n)
    s_linear = cvx.Variable(n)
    c1 = cvx.Parameter(value=c1, nonneg=True)
    c2 = cvx.Parameter(value=c2, nonneg=True)
    c3 = cvx.Parameter(value=c3, nonneg=True)
    tau = cvx.Parameter(value=tau)
    # w = len(signal) / np.sum(index_set)
    beta = cvx.Variable()
    objective = cvx.Minimize(
        # (365 * 3 / len(signal)) * w * cvx.sum(0.5 * cvx.abs(s_error) + (tau - 0.5) * s_error)
        2 * cvx.sum(0.5 * cvx.abs(cvx.multiply(residual_weights, s_error))
                    + (tau - 0.5) * cvx.multiply(residual_weights, s_error))
        + c1 * cvx.norm1(cvx.multiply(tv_weights, cvx.diff(s_hat, k=1)))
        + c2 * cvx.norm(cvx.diff(s_seas, k=2))
        + c3 * beta ** 2
    )
    constraints = [
        signal[use_ixs] == s_hat[use_ixs] + s_seas[use_ixs] + s_error[use_ixs],
        cvx.sum(s_seas[:365]) == 0
    ]
    if len(signal) > 365:
        constraints.append(s_seas[365:] - s_seas[:-365] == beta)
        constraints.extend([beta <= 0.01, beta >= -0.1])
    problem = cvx.Problem(objective=objective, constraints=constraints)
    problem.solve(solver='MOSEK')
    return s_hat.value, s_seas.value

def basic_outlier_filter(x, outlier_constant=1.5):
    '''
    Applies an outlier filter based on the interquartile range definition:
        any data point more than 1.5 interquartile ranges (IQRs) below the
        first quartile or above the third quartile

    Function returns a boolean mask for entries in the input array that are
    not outliers.

    :param x: ndarray
    :param outlier_constant: float, multiplier constant on IQR
    :return: boolean mask
    '''
    a = np.array(x)
    upper_quartile = np.percentile(a, 75)
    lower_quartile = np.percentile(a, 25)
    iqr = (upper_quartile - lower_quartile) * outlier_constant
    quartile_set = (lower_quartile - iqr, upper_quartile + iqr)
    mask = np.logical_and(
        a >= quartile_set[0],
        a <= quartile_set[1]
    )
    return mask

def progress(count, total, status='', bar_length=60):
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
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()

def find_runs(x):
    """Find runs of consecutive items in an array.
    https://gist.github.com/alimanfoo/c5977e87111abe8127453b21204c1065"""

    # ensure array
    x = np.asanyarray(x)
    if x.ndim != 1:
        raise ValueError('only 1D array supported')
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
