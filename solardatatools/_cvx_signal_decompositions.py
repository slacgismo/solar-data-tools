import sys
import numpy as np

import cvxpy as cvx


def _cvx_l2_l1d1_l2d2p365(
    signal,
    use_ixs=None,
    w0=10, # "hard-coded"
    w1=50, # optimized
    w2=1e5,
    yearly_periodic=False,
    return_all=False, 
    solver="MOSEK",
    transition_locs=None,
    verbose=False,
):
    """
    This performs total variation filtering with the addition of a seasonal
    baseline fit. This introduces a new signal to the model that is smooth and
    periodic on a yearly time frame. This does a better job of describing real,
    multi-year solar PV power data sets, and therefore does an improved job of
    estimating the discretely changing signal.

    :param signal: A 1d numpy array (must support boolean indexing) containing
    the signal of interest
    :param w0: Weight on the residual component
    :param w1: The regularization parameter to control the total variation in
    the final output signal
    :param w2: The regularization parameter to control the smoothness of the
    seasonal signal
    :return: A 1d numpy array containing the filtered signal
    """
    if use_ixs is None:
        index_set = ~np.isnan(signal)
    else:
        index_set = np.logical_and(use_ixs, ~np.isnan(signal))

    # Iterative reweighted L1 heuristic
    tv_weights = np.ones(len(signal) - 1)
    eps = 0.1
    n_iter = 5

    w0 = cvx.Constant(value=w0)
    w1 = cvx.Constant(value=w1)
    w2 = cvx.Constant(value=w2)

    for i in range(n_iter):
        s_hat = cvx.Variable(len(signal))
        s_seas = cvx.Variable(len(signal))
        s_error = cvx.Variable(len(signal))

        if transition_locs is None: # TODO: this should be two separate sd problems
            objective = cvx.Minimize(
                w0 * cvx.sum_squares(s_error)
                + w1 * cvx.norm1(cvx.multiply(tv_weights, cvx.diff(s_hat, k=1)))
                + w2 * cvx.sum_squares(cvx.diff(s_seas, k=2))
            )
        else:
            objective = cvx.Minimize(
                w0 * cvx.norm(s_error)
               + w2 * cvx.sum_squares(cvx.diff(s_seas, k=2))
            )
        # Consistency constraints
        constraints = [
            signal[index_set] == s_hat[index_set] + s_seas[index_set] + s_error[index_set],
            cvx.sum(s_seas[:365]) == 0,
        ]
        if len(signal) > 365:
            constraints.append(s_seas[365:] - s_seas[:-365] == 0)
            if yearly_periodic:
                constraints.append(s_hat[365:] - s_hat[:-365] == 0)
        if transition_locs is not None:
            loc_mask = np.ones(len(signal) - 1, dtype=bool)
            loc_mask[transition_locs] = False
            constraints.append(cvx.diff(s_hat, k=1)[loc_mask] == 0)

        problem = cvx.Problem(objective=objective, constraints=constraints)
        problem.solve(solver=solver, verbose=verbose)

        tv_weights = 1 / (eps + np.abs(np.diff(s_hat.value, n=1)))

    if return_all:
        return s_hat.value, s_seas.value, s_error.value, problem.objective.value

    return s_hat.value, s_seas.value


def _cvx_tl1_l2d2p365(
    signal,
    use_ixs=None,
    tau=0.75, # passed as 0.05/0.1 (sunrise), 0.95/0.9 (sunset)
    w0=1,
    w1=500,
    yearly_periodic=True,
    return_all=False,
    solver="MOSEK",
    verbose=False
):
    """
    https://colab.research.google.com/github/cvxgrp/cvx_short_course/blob/master/applications/quantile_regression.ipynb

    :param signal: 1d numpy array
    :param use_ixs: optional index set to apply cost function to
    :param tau: float, parameter for quantile regression
    :param c1: float
    :param solver: string
    :return: median fit with seasonal baseline removed
    """
    if use_ixs is None:
        use_ixs = ~np.isnan(signal)
    x = cvx.Variable(len(signal))
    r = signal[use_ixs] - x[use_ixs]
    objective = cvx.Minimize(
        w0 * cvx.sum(0.5 * cvx.abs(r) + (tau - 0.5) * r) +
        w1 * cvx.sum_squares(cvx.diff(x, k=2))
    )
    if len(signal) > 365 and yearly_periodic:
        constraints = [x[365:] == x[:-365]]
    else:
        constraints = []
    problem = cvx.Problem(objective, constraints=constraints)
    problem.solve(solver=solver, verbose=verbose)

    if return_all:
        return x.value, problem.objective.value

    return x.value


def _cvx_l1_l1d1_l2d2p365(
    signal,
    use_ixs=None,
    w0=3,  # l1 term
    w1=18, # l1d1 term
    w2=6000, # seasonal term
    w3=300, # linear term
    return_all=False,
    solver="MOSEK",
    verbose=False
):
    """
    This performs total variation filtering with the addition of a seasonal baseline fit. This introduces a new
    signal to the model that is smooth and periodic on a yearly time frame. This does a better job of describing real,
    multi-year solar PV power data sets, and therefore does an improved job of estimating the discretely changing
    signal.

    :param signal: A 1d numpy array (must support boolean indexing) containing the signal of interest
    :param w0: Weight on the residual component
    :param w1: The regularization parameter to control the total variation in the final output signal
    :param w2: The regularization parameter to control the smoothness of the seasonal signal
    :param w3: The regularization parameter to control the degradation slope of the seasonal signal
    :return: A 1d numpy array containing the filtered signal
    """
    n = len(signal)

    if use_ixs is None:
        use_ixs = ~np.isnan(signal)
    else:
        use_ixs = np.logical_and(use_ixs, ~np.isnan(signal))

    w0 = cvx.Parameter(value=w0, nonneg=True)
    w1 = cvx.Parameter(value=w1, nonneg=True)
    w2 = cvx.Parameter(value=w2, nonneg=True)
    w3 = cvx.Parameter(value=w3, nonneg=True)

    # Iterative reweighted L1 heuristic
    tv_weights = np.ones(n - 1)
    eps = 0.1
    n_iter = 5

    for i in range(n_iter):
        s_hat = cvx.Variable(n)
        s_seas = cvx.Variable(max(n, 366))
        s_error = cvx.Variable(n)

        beta = cvx.Variable()

        objective = cvx.Minimize(
              w0 * cvx.sum(0.5 * cvx.abs(s_error))
            + w1 * cvx.norm1(cvx.multiply(tv_weights, cvx.diff(s_hat, k=1)))
            + w2 * cvx.sum_squares(cvx.diff(s_seas, k=2))
            + w3 * beta**2 # linear term that has a slope of beta over 1 year
        )

        constraints = [
            signal[use_ixs] == s_hat[use_ixs] + s_seas[:n][use_ixs] + s_error[use_ixs],
            cvx.sum(s_seas[:365]) == 0,
        ]
        constraints.append(s_seas[365:] - s_seas[:-365] == beta)
        constraints.extend([beta <= 0.01, beta >= -0.1])
        problem = cvx.Problem(objective=objective, constraints=constraints)
        problem.solve(solver=solver, verbose=verbose)

        tv_weights = 1 / (eps + np.abs(np.diff(s_hat.value, n=1)))

    if return_all:
        return s_hat.value, s_seas.value, s_error.value, problem.objective.value

    return s_hat.value, s_seas.value[:n], beta.value**2


def _cvx_make_l2_l1d2_constrained(
        signal,
        w1=1e1,
        return_all=False,
        solver="MOSEK",
        verbose=False
):
    """
    Used in solardatatools/algorithms/clipping.py
    Added hard-coded constraints on the first and last vals
    """
    use_ixs = ~np.isnan(signal)

    y_hat = cvx.Variable(len(signal))
    mu = cvx.Parameter(nonneg=True)
    mu.value = w1
    error = cvx.sum_squares(signal[use_ixs] - y_hat[use_ixs])
    reg = cvx.norm(cvx.diff(y_hat, k=2), p=1)

    objective = cvx.Minimize(error + mu * reg)
    constraints = [y_hat[0] == 0, y_hat[-1] == 1]
    problem = cvx.Problem(objective, constraints)
    problem.solve(solver=solver, verbose=verbose)

    if return_all:
        return y_hat.value, problem.objective.value

    return problem, signal, y_hat.value, mu.value
