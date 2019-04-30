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

def total_variation_plus_seasonal_filter(signal, c1=10, c2=500):
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
    s_hat = cvx.Variable(len(signal))
    s_seas = cvx.Variable(len(signal))
    s_error = cvx.Variable(len(signal))
    c1 = cvx.Constant(value=c1)
    c2 = cvx.Constant(value=c2)
    index_set = ~np.isnan(signal)
    w = len(signal) / np.sum(index_set)
    objective = cvx.Minimize(
        (365 * 3 / len(signal)) * w * cvx.sum(cvx.huber(s_error))
        + c1 * cvx.norm1(cvx.diff(s_hat, k=1))
        + c2 * cvx.norm(cvx.diff(s_seas, k=2))
        + c2 * .1 * cvx.norm(cvx.diff(s_seas, k=1))
    )
    constraints = [
        signal[index_set] == s_hat[index_set] + s_seas[index_set] + s_error[index_set],
        s_seas[365:] - s_seas[:-365] == 0,
        cvx.sum(s_seas[:365]) == 0
    ]
    problem = cvx.Problem(objective=objective, constraints=constraints)
    problem.solve()
    return s_hat.value, s_seas.value

def local_median_regression_with_seasonal(signal, c1=1e3, solver='ECOS'):
    '''
    for a list of available solvers, see:
        https://www.cvxpy.org/tutorial/advanced/index.html#solve-method-options

    :param signal: 1d numpy array
    :param c1: float
    :param solver: string
    :return: median fit with seasonal baseline removed
    '''
    x = cvx.Variable(len(signal))
    objective = cvx.Minimize(
        cvx.norm1(signal - x) + c1 * cvx.norm(cvx.diff(x, k=2))
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