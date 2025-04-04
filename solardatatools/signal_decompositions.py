# -*- coding: utf-8 -*-
"""
Signal Decompositions Module

This module contains standardized signal decomposition models for use in the
SDT algorithms. The defined signal decompositions are:

1. `l2_l1d1_l2d2p365`: separating a piecewise constant component from a smooth
   and seasonal component, with Gaussian noise
   - l2: gaussian noise, sum-of-squares small or l2-norm squared
   - l1d1: piecewise constant heuristic, l1-norm of first order differences
   - l2d2p365: small second order diffs (smooth) and 365-periodic

2. `tl1_l2d2p365`: similar to (2), estimating a smooth, seasonal component with
   an asymmetric laplacian noise model, fitting a local quantile instead of a local average
   - tl1: 'tilted l1-norm,' also known as quantile cost function
   - l2d2p365: small second order diffs (smooth) and 365-periodic

3. `l1_l1d1_l2d2p365`: like (1) but with an asymmetric residual cost instead
   of Gaussian residuals
   - l1: l1-norm
   - l1d1: piecewise constant heuristic, l1-norm of first order differences
   - l2d2p365: small second order diffs (smooth) and 365-periodic

4. `l2_l1d2_constrained`:
   - l2: gaussian noise, sum-of-squares small or l2-norm squared
   - l1d2: piecewise linear heuristic
   - constrained to have first val at 0 and last val at 1

"""

from solardatatools._osd_signal_decompositions import (
    _osd_l2_l1d1_l2d2p365,
    _osd_tl1_l2d2p365,
    _osd_l2_l1d2_constrained,
)
from solardatatools._cvx_signal_decompositions import (
    _cvx_l2_l1d1_l2d2p365,
    _cvx_l1_pwc_smoothper_trend,
    _cvx_tl1_l2d2p365,
    _cvx_l2_l1d2_constrained,
)


def l2_l1d1_l2d2p365(
    signal,
    use_ixs=None,
    w1=50,
    w2=1e-3,
    yearly_periodic=False,
    return_all=False,  # for unit tests only
    solver="CLARABEL",
    sum_card=False,  # OSD only
    transition_locs=None,  # CVX only
    verbose=False,
):
    """
    Used in: solardatatools/algorithms/time_shifts.py

    This performs total variation filtering with the addition of a seasonal
    baseline fit. This introduces a new signal to the model that is smooth and
    periodic on a yearly time frame. This does a better job of describing real,
    multi-year solar PV power data sets, and therefore does an improved job of
    estimating the discretely changing signal. Default solver is QSS and timeshift
    algorithm takes the final solution from solving the nonconvex problem
    with sum_card=True.

    :param signal: A 1d numpy array (must support boolean indexing) containing
        the signal of interest
    :param w0: Weight on the residual component
    :param w1: The regularization parameter to control the total variation in
        the final output signal
    :param w2: The regularization parameter to control the smoothness of the
        seasonal signal
    :param yearly_periodic: Adds periodicity constraint to signal decomposition
    :param return_all: Returns all components and the objective value. Used for tests.
    :param solver: Solver to use for the decomposition. Supported solvers are CLARABEL and MOSEK.
    :param sum_card: Boolean for using the nonconvex formulation using the cardinality penalty,
        Supported only using OSD with the QSS solver.
    :param transition_locs: List of indices where transitions are located. Only used in CVXPY problems.
    :param verbose: Sets verbosity
    :return: A tuple with two 1d numpy arrays containing the two signal component estimates
    """
    if solver == "MOSEK":
        res = _cvx_l2_l1d1_l2d2p365(
            signal=signal,
            use_ixs=use_ixs,
            w0=1,
            w1=w1,
            w2=w2,
            yearly_periodic=yearly_periodic,
            return_all=return_all,
            transition_locs=transition_locs,
            solver=solver,
            verbose=verbose,
        )
    elif yearly_periodic:
        res = _osd_l2_l1d1_l2d2p365(
            signal=signal,
            use_ixs=use_ixs,
            w1=w1,
            w2=w2,
            yearly_periodic=yearly_periodic,
            return_all=return_all,
            transition_locs=transition_locs,
            solver=solver,
            sum_card=False,
            verbose=verbose,
        )
    else:
        res = _osd_l2_l1d1_l2d2p365(
            signal=signal,
            use_ixs=use_ixs,
            w1=w1,
            w2=w2,
            yearly_periodic=yearly_periodic,
            return_all=return_all,
            transition_locs=transition_locs,
            solver=solver,
            sum_card=False,
            verbose=verbose,
        )

    return res


def tl1_l2d2p365(
    signal,
    use_ixs=None,
    tau=0.75,
    w0=1,
    yearly_periodic=True,
    return_all=False,
    solver="CLARABEL",
    verbose=False,
):
    """
    Used in:
    solardatatools/algorithms/sunrise_sunset_estimation.py
    solardatatools/clear_day_detection.py
    solardatatools/data_quality.py
    solardatatools/sunrise_sunset.py


    This is a convex problem and the default solver across SDT is CLARABEL.

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
    :param solver: Solver to use for the decomposition. Supported solvers are CLARABEL and MOSEK.
    :param verbose: Sets verbosity
    :return: A tuple with three 1d numpy arrays containing the three signal component estimates
    """
    if solver == "MOSEK":
        res = _cvx_tl1_l2d2p365(
            signal=signal,
            use_ixs=use_ixs,
            tau=tau,
            w0=w0,
            yearly_periodic=yearly_periodic,
            return_all=return_all,
            solver=solver,
            verbose=verbose,
        )
    else:
        res = _osd_tl1_l2d2p365(
            signal=signal,
            use_ixs=use_ixs,
            tau=tau,
            w0=w0,
            return_all=return_all,
            solver=solver,
            verbose=verbose,
        )

    return res


def l1_pwc_smoothper_trend(
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
    :param solver: Solver to use for the decomposition. Supported solvers are CLARABEL and MOSEK.
    :param verbose: Sets verbosity
    :return: A tuple with three 1d numpy arrays containing the three non-noise signal component estimates
    """
    res = _cvx_l1_pwc_smoothper_trend(
        signal=signal,
        use_ixs=use_ixs,
        w2=w2,
        w3=w3,
        w4=w4,
        solver=solver,
        verbose=verbose,
        return_all=return_all,
    )
    return res


def l2_l1d2_constrained(
    signal, w0=1, w1=5, return_all=False, solver="CLARABEL", verbose=False
):
    """
    Used in solardatatools/algorithms/clipping.py

    This is a convex problem and the default solver across SDT is CLARABEL.

    :param signal: A 1d numpy array (must support boolean indexing) containing
        the signal of interest
    :param w0: Weight on the residual component
    :param w1: The regularization parameter on l1d2 component
    :param return_all: Returns all components and the objective value. Used for tests.
    :param solver: Solver to use for the decomposition. Supported solvers are CLARABEL and MOSEK.
    :param verbose: Sets verbosity
    :return: A tuple with returning the signal, the l1d2 component estimate, and the weight
    """
    if solver == "MOSEK":
        # MOSEK weights set in CVXPY function
        res = _cvx_l2_l1d2_constrained(
            signal, return_all=return_all, solver=solver, verbose=verbose
        )
    else:
        res = _osd_l2_l1d2_constrained(
            signal, w0=w0, w1=w1, return_all=return_all, solver=solver, verbose=verbose
        )

    return res
