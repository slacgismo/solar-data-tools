"""
Signal Decompositions Module for CVXPY

This module contains standardized signal decomposition models for use in the SDT algorithms using CVXPY and the MOSEK solver. The defined signal decompositions are:

1. `_cvx_l2_l1d1_l2d2p365`: Separating a piecewise constant component from a smooth and seasonal component, with Gaussian noise
   - `l2`: Gaussian noise, sum-of-squares small or l2-norm squared
   - `l1d1`: Piecewise constant heuristic, l1-norm of first-order differences
   - `l2d2p365`: Small second-order differences (smooth) and 365-periodic

2. `_cvx_tl1_l2d2p365`: Similar to (2), estimating a smooth, seasonal component with an asymmetric Laplacian noise model, fitting a local quantile instead of a local average
   - `tl1`: 'Tilted l1-norm,' also known as quantile cost function
   - `l2d2p365`: Small second-order differences (smooth) and 365-periodic

3. `_cvx_l1_l1d1_l2d2p365`: Like (1) but with an asymmetric residual cost instead of Gaussian residuals
   - `l1`: l1-norm
   - `l1d1`: Piecewise constant heuristic, l1-norm of first-order differences
   - `l2d2p365`: Small second-order differences (smooth) and 365-periodic

4. `_cvx_l2_l1d2_constrained`:
   - `l2`: Gaussian noise, sum-of-squares small or l2-norm squared
   - `l1d2`: Piecewise linear heuristic
   - Constrained to have first value at 0 and last value at 1

"""

import sys
import numpy as np

import cvxpy as cvx
from spcqe import make_basis_matrix, make_regularization_matrix


def _cvx_l2_l1d1_l2d2p365(
    signal,
    use_ixs=None,
    w0=10,  # "hard-coded"
    w1=50,  # optimized
    w2=1e5,
    yearly_periodic=False,
    return_all=False,
    solver="MOSEK",
    transition_locs=None,
    verbose=False,
):
    """
    Used in: solardatatools/algorithms/time_shifts.py

    This performs total variation filtering with the addition of a seasonal
    baseline fit. This introduces a new signal to the model that is smooth and
    periodic on a yearly time frame. This does a better job of describing real,
    multi-year solar PV power data sets, and therefore does an improved job of
    estimating the discretely changing signal.

    Parameters
    ----------
    signal : array_like
        A 1d numpy array (must support boolean indexing) containing
        the signal of interest.
    w0 : float
        Weight on the residual component.
    w1 : float
        The regularization parameter to control the total variation in
        the final output signal.
    w2 : float
        The regularization parameter to control the smoothness of the
        seasonal signal.
    yearly_periodic : bool, optional
        Adds periodicity constraint to signal decomposition.
    return_all : bool, optional
        Returns all components and the objective value. Used for tests.
    solver : str, optional
        Solver to use for the decomposition.
    transition_locs : list of int, optional
        List of indices where transitions are located.
    verbose : bool, optional
        Sets verbosity.

    Returns
    -------
    tuple of array_like
        A tuple with two 1d numpy arrays containing the two signal component estimates.
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

        if transition_locs is None:  # TODO: this should be two separate sd problems
            objective = cvx.Minimize(
                w0 * cvx.sum_squares(s_error)
                + w1 * cvx.norm1(cvx.multiply(tv_weights, cvx.diff(s_hat, k=1)))
                + w2 * cvx.sum_squares(cvx.diff(s_seas, k=2))
            )
        else:
            objective = cvx.Minimize(
                w0 * cvx.norm(s_error) + w2 * cvx.sum_squares(cvx.diff(s_seas, k=2))
            )
        # Consistency constraints
        constraints = [
            signal[index_set]
            == s_hat[index_set] + s_seas[index_set] + s_error[index_set],
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
    tau=0.75,  # passed as 0.05/0.1 (sunrise), 0.95/0.9 (sunset)
    w0=1,
    w1=500,
    yearly_periodic=True,
    return_all=False,
    solver="MOSEK",
    verbose=False,
):
    """
    Used in:
        solardatatools/algorithms/sunrise_sunset_estimation.py
        solardatatools/clear_day_detection.py
        solardatatools/data_quality.py
        solardatatools/sunrise_sunset.py

    :param signal: A 1d numpy array (must support boolean indexing) containing
        the signal of interest
    :param use_ixs: List of booleans indicating indices to use in signal.
        None is default (uses the entire signal).
    :param tau: Quantile regression parameter,between zero and one, and it sets
        the approximate quantile of the residual distribution that the model is fit to
        See: https://colab.research.google.com/github/cvxgrp/cvx_short_course/blob/master/applications/quantile_regression.ipynb
    :param w0: Weight on the residual component
    :param w1: The regularization parameter to control the smoothness of the
        seasonal signal
    :param yearly_periodic: Adds periodicity constraint to signal decomposition
    :param return_all: Returns all components and the objective value. Used for tests.
    :param solver: Solver to use for the decomposition
    :param verbose: Sets verbosity
    :return: A tuple with three 1d numpy arrays containing the three signal component estimates
    """
    if use_ixs is None:
        use_ixs = ~np.isnan(signal)
    x = cvx.Variable(len(signal))
    r = signal[use_ixs] - x[use_ixs]
    objective = cvx.Minimize(
        w0 * cvx.sum(0.5 * cvx.abs(r) + (tau - 0.5) * r)
        + w1 * cvx.sum_squares(cvx.diff(x, k=2))
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


def _cvx_l1_pwc_smoothper_trend(
    signal,
    use_ixs=None,
    w2=2e1,
    w3=1,
    w4=1e1,
    solver="CLARABEL",
    verbose=False,
    return_all=False,
):
    """
    Used in solardatatools/algorithms/capacity_change.py

    We solve a convex signal decomposition problem, making use of the l1-sparsity heuristic for estimating a
    piecewise constant component. We use a single pass of iterative reweighting on this term to "polish" the sparse
    solution (see: https://web.stanford.edu/~boyd/papers/rwl1.html).

    :param signal: A 1d numpy array (must support boolean indexing) containing
        the signal of interest
    :param use_ixs: List of booleans indicating indices to use in signal.
        None is default (uses the entire signal).
    :param w2: Weight on the piecewise constant component
    :param w3: Weight on the smooth, periodic component
    :param w4: Weight on the slope of the trend term (discourages large trends)
    :param solver: Solver to use for the decomposition. Standard cvxpy solvers are supported
    :param verbose: Sets verbosity
    :return: A tuple with three 1d numpy arrays containing the three non-noise signal component estimates
    """
    if solver is not None:
        if solver.upper() not in ["MOSEK", "CLARABEL"]:
            print("Only CLARABEL and MOSEK supported. Using CLARABEL...")
            solver = "CLARABEL"
    masked_sig = np.copy(signal)
    if use_ixs is not None:
        masked_sig[~use_ixs] = np.nan
    problem, tv_weights_param = make_l1_pwc_smoothper_trend_problem(
        masked_sig, w2, w3, w4
    )
    problem.solve(solver=solver, verbose=verbose)
    var_dict = {v.name(): v for v in problem.variables()}
    eps = 0.1
    tv_weights = 1 / (eps + np.abs(np.diff(var_dict["x2"].value, n=1)))
    tv_weights_param.value = tv_weights
    problem.solve(solver=solver, verbose=verbose)
    var_dict = {v.name(): v for v in problem.variables()}
    if not return_all:
        return var_dict["x2"].value, var_dict["x3"].value, var_dict["x4"].value
    else:
        return (
            var_dict["x2"].value,
            var_dict["x3"].value,
            var_dict["x4"].value,
            problem,
        )


def make_l1_pwc_smoothper_trend_problem(metric, w2, w3, w4, tv_weights=None):
    use_set = ~np.isnan(metric)
    # noise term
    x1 = cvx.Variable(len(metric), name="x1")
    # piecewise constatn term
    x2 = cvx.Variable(len(metric), name="x2")
    # Smooth, yearly periodic term
    x3 = cvx.Variable(len(metric), name="x3")
    # trend term
    x4 = cvx.Variable(len(metric), name="x4")
    phi1 = cvx.mean(cvx.abs(x1))
    # we parameterize this weight for iterative reweighted L1
    if tv_weights is None:
        tv_weights = np.ones(len(metric) - 1)
    tv_weights_param = cvx.Parameter(
        shape=len(tv_weights), value=tv_weights, nonneg=True
    )
    phi2 = w2 * cvx.mean(cvx.abs(cvx.multiply(tv_weights_param, cvx.diff(x2, k=1))))
    B = make_basis_matrix(num_harmonics=[6], length=len(metric), periods=[365.2425])
    W = make_regularization_matrix(
        num_harmonics=[6], weight=w3, periods=[365.2425]
    ).todense()
    # remove bias term
    B = B[:, 1:]
    W = W[1:, 1:]
    z3 = cvx.Variable(B.shape[1], name="z3")
    phi3 = cvx.sum_squares(W @ z3)
    beta = cvx.Variable(name="beta")
    phi4 = w4 * len(metric) * beta**2
    cost = phi1 + phi2 + phi3 + phi4
    constraints = [
        metric[use_set] == (x1 + x2 + x3 + x4)[use_set],
        x3 == B @ z3,
        cvx.diff(x4) == beta,
        beta * 365 <= 0.05,
        beta * 365 >= -0.2,
        x4[0] == 0,
    ]
    problem = cvx.Problem(cvx.Minimize(cost), constraints)
    return problem, tv_weights_param


def _cvx_l2_l1d2_constrained(
    signal, w1=1e1, return_all=False, solver="MOSEK", verbose=False
):
    """
    Used in solardatatools/algorithms/clipping.py

    This is a convex problem and the default solver across SDT is OSQP.

    :param signal: A 1d numpy array (must support boolean indexing) containing
        the signal of interest
    :param w0: Weight on the residual component
    :param w1: The regularization parameter on l1d2 component
    :param return_all: Returns all components and the objective value. Used for tests.
    :param solver: Solver to use for the decomposition
    :param verbose: Sets verbosity
    :return: A tuple with returning the signal, the l1d2 component estimate, and the weight
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

    return signal, y_hat.value, mu.value
