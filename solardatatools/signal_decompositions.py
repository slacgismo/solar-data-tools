# -*- coding: utf-8 -*-
''' Signal Decompositions Module

This module contains standardized signal decomposition models for use in the
SDT algorithms. The defined signal decompositions are:

1) 'l2_l1d1_l2d2p365': separating a piecewise constant component from a smooth
and seasonal component, with Gaussian noise
    - l2: gaussian noise, sum-of-squares small or l2-norm squared
    - l1d1: piecewise constant heuristic, l1-norm of first order differences
    - l2d2p365: small second order diffs (smooth) and 365-periodic
2) 'l1_l2d2p365': estimating a smooth, seasonal component with a laplacian
noise model, fitting a local median instead of a local average
    - l1: laplacian noise, sum-of-absolute values or l1-norm
    - l2d2p365: small second order diffs (smooth) and 365-periodic
3) 'tl1_l2d2p365': similar to (2), estimating a smooth, seasonal component with
an asymmetric laplacian noise model, fitting a local quantile instead of a
local average
    - tl1: 'tilted l1-norm,' also known as quantile cost function
    - l2d2p365: small second order diffs (smooth) and 365-periodic
4) 'tl1_l1d1_l2d2p365': like (1) but with an asymmetric residual cost instead
of Gaussian residuals
    - tl1: 'tilted l1-norm,' also known as quantile cost function
    - l1d1: piecewise constant heuristic, l1-norm of first order differences
    - l2d2p365: small second order diffs (smooth) and 365-periodic
5) 'hu_l1d1': total variation denoising with Huber residual cost
    - hu: Huber cost, a function that is quadratic below a cutoff point and
    linear above the cutoff point
    - l1d1: piecewise constant heuristic, l1-norm of first order differences

'''
import sys
import numpy as np
import cvxpy as cvx


def l2_l1d1_l2d2p365(signal, c1=10, c2=500,
                     solver=None, verbose=False,
                     residual_weights=None, tv_weights=None,
                     use_ixs=None, yearly_periodic=False,
                     transition_locs=None,
                     seas_max=None):
    '''
    This performs total variation filtering with the addition of a seasonal
    baseline fit. This introduces a new signal to the model that is smooth and
    periodic on a yearly time frame. This does a better job of describing real,
    multi-year solar PV power data sets, and therefore does an improved job of
    estimating the discretely changing signal.

    :param signal: A 1d numpy array (must support boolean indexing) containing
    the signal of interest
    :param c1: The regularization parameter to control the total variation in
    the final output signal
    :param c2: The regularization parameter to control the smoothness of the
    seasonal signal
    :return: A 1d numpy array containing the filtered signal
    '''
    if residual_weights is None:
        residual_weights = np.ones_like(signal)
    if tv_weights is None:
        tv_weights = np.ones(len(signal) - 1)
    if use_ixs is None:
        index_set = ~np.isnan(signal)
    else:
        index_set = np.logical_and(use_ixs, ~np.isnan(signal))
    s_hat = cvx.Variable(len(signal))
    s_seas = cvx.Variable(len(signal))
    s_error = cvx.Variable(len(signal))
    c1 = cvx.Constant(value=c1)
    c2 = cvx.Constant(value=c2)
    #w = len(signal) / np.sum(index_set)
    if transition_locs is None:
        objective = cvx.Minimize(
            # (365 * 3 / len(signal)) * w *
            # cvx.sum(cvx.huber(cvx.multiply(residual_weights, s_error)))
            10 * cvx.norm(cvx.multiply(residual_weights, s_error))
            + c1 * cvx.norm1(cvx.multiply(tv_weights, cvx.diff(s_hat, k=1)))
            + c2 * cvx.norm(cvx.diff(s_seas, k=2))
            # + c2 * .1 * cvx.norm(cvx.diff(s_seas, k=1))
        )
    else:
        objective = cvx.Minimize(
            10 * cvx.norm(cvx.multiply(residual_weights, s_error))
            + c2 * cvx.norm(cvx.diff(s_seas, k=2))
        )
    constraints = [
        signal[index_set] == s_hat[index_set] + s_seas[index_set] + s_error[index_set],
        cvx.sum(s_seas[:365]) == 0
    ]
    if len(signal) > 365:
        constraints.append(s_seas[365:] - s_seas[:-365] == 0)
        if yearly_periodic:
            constraints.append(s_hat[365:] - s_hat[:-365] == 0)
    if transition_locs is not None:
        loc_mask = np.ones(len(signal) - 1, dtype=bool)
        loc_mask[transition_locs] = False
        # loc_mask[transition_locs + 1] = False
        constraints.append(cvx.diff(s_hat, k=1)[loc_mask] == 0)
    if seas_max is not None:
        constraints.append(s_seas <= seas_max)
    problem = cvx.Problem(objective=objective, constraints=constraints)
    problem.solve(solver=solver, verbose=verbose)
    return s_hat.value, s_seas.value

def l1_l2d2p365(signal, use_ixs=None, c1=1e3, yearly_periodic=True,
                solver=None, verbose=False):
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
    if len(signal) > 365 and yearly_periodic:
        constraints = [
            x[365:] == x[:-365]
        ]
    else:
        constraints = []
    prob = cvx.Problem(objective, constraints=constraints)
    # Currently seems to work with SCS or MOSEK
    prob.solve(solver=solver, verbose=verbose)
    return x.value

def tl1_l2d2p365(signal, use_ixs=None, tau=0.75,
                 c1=1e3, solver=None,
                 yearly_periodic=True,
                 verbose=False,
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
    if len(signal) > 365 and yearly_periodic:
        constraints = [
            x[365:] == x[:-365]
        ]
    else:
        constraints = []
    prob = cvx.Problem(objective, constraints=constraints)
    prob.solve(solver=solver, verbose=verbose)
    return x.value


def tl1_l1d1_l2d2p365(signal, use_ixs=None, tau=0.995,
                      c1=1e3, c2=1e2, c3=1e2,
                      solver=None, verbose=False,
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
    s_seas = cvx.Variable(max(n, 366))
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
        signal[use_ixs] == s_hat[use_ixs] + s_seas[:n][use_ixs] + s_error[use_ixs],
        cvx.sum(s_seas[:365]) == 0
    ]
    if True:
        constraints.append(s_seas[365:] - s_seas[:-365] == beta)
        constraints.extend([beta <= 0.01, beta >= -0.1])
    problem = cvx.Problem(objective=objective, constraints=constraints)
    problem.solve(solver=solver, verbose=verbose)
    return s_hat.value, s_seas.value[:n]


def make_l2_ll1d1(y, weight=1e1):
    y_hat = cvx.Variable(len(y))
    y_param = cvx.Parameter(len(y), value=y)
    mu = cvx.Parameter(nonneg=True)
    mu.value = weight
    error = cvx.sum_squares(y_param - y_hat)
    reg = cvx.norm(cvx.diff(y_hat, k=2), p=1)
    objective = cvx.Minimize(error + mu * reg)
    constraints = [
        y_param[0] == y_hat[0],
        y[-1] == y_hat[-1]
    ]
    problem = cvx.Problem(objective, constraints)
    return problem, y_param, y_hat, mu

##############################################################################
# NOT CURRENTLY USED
##############################################################################

def hu_l1d1(signal, C=5):
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