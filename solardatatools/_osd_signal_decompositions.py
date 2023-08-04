""" Signal Decompositions Module for OSD

This module contains standardized signal decomposition models for use in the
SDT algorithms using the OSD modeling language. The defined signal decompositions are:

1) '_osd_l2_l1d1_l2d2p365': separating a piecewise constant component from a smooth
and seasonal component, with Gaussian noise
    - l2: gaussian noise, sum-of-squares small or l2-norm squared
    - l1d1: piecewise constant heuristic, l1-norm of first order differences
    - l2d2p365: small second order diffs (smooth) and 365-periodic
2) '_osd_tl1_l2d2p365': similar to (2), estimating a smooth, seasonal component with
an asymmetric laplacian noise model, fitting a local quantile instead of a
local average
    - tl1: 'tilted l1-norm,' also known as quantile cost function
    - l2d2p365: small second order diffs (smooth) and 365-periodic
3) '_osd_l1_l1d1_l2d2p365': like (1) but with an asymmetric residual cost instead
of Gaussian residuals
    - l1: l1-norm
    - l1d1: piecewise constant heuristic, l1-norm of first order differences
    - l2d2p365: small second order diffs (smooth) and 365-periodic
4) '_osd_l2_l1d2_constrained':
    - l2: gaussian noise, sum-of-squares small or l2-norm squared
    - l1d2: piecewise linear heuristic
    - constrained to have first val at 0 and last val at 1
"""
import sys
import numpy as np

from gfosd import Problem
from gfosd.components import SumAbs, SumSquare, SumCard, SumQuantile, Aggregate, AverageEqual,\
    Periodic, Inequality, FirstValEqual, LastValEqual, NoCurvature, NoSlope


def _osd_l2_l1d1_l2d2p365(
        signal,
        w0=10,
        w1=50,
        w2=1e5,
        return_all=False,
        yearly_periodic=False,
        solver="QSS",
        use_ixs=None,
        sum_card=False,
        verbose=False
):
    """
    Used in: solardatatools/algorithms/time_shifts.py

    This performs total variation filtering with the addition of a seasonal
    baseline fit. This introduces a new signal to the model that is smooth and
    periodic on a yearly time frame. This does a better job of describing real,
    multi-year solar PV power data sets, and therefore does an improved job of
    estimating the discretely changing signal. Default solver is QSS, and timeshift
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
    :param solver: Solver to use for the decomposition
    :param sum_card: Boolean for using the nonconvex formulation using the cardinality penalty,
    Supported only using OSD with the QSS solver.
    :param verbose: Sets verbosity
    :return: A tuple with two 1d numpy arrays containing the two signal component estimates
    """
    if solver != "QSS":
        sum_card=False

    if sum_card:
        # Scale objective
        w0 /= 1e6
        w1 /= 1e6
        w2 /= 1e6
    elif solver == "QSS":
        # Scale objective
        w0 /= 1e4
        w1 /= 1e4
        w2 /= 1e4

    c1 = SumSquare(weight=w0)

    c2 = SumSquare(weight=w2, diff=2)

    if sum_card:
        c3 = SumCard(weight=w1, diff=1)
    else:
        c3 = SumAbs(weight=w1, diff=1)

    if len(signal) >= 365:
        c2 = Aggregate([SumSquare(weight=w2, diff=2), AverageEqual(0, period=365), Periodic(365)])
        if yearly_periodic and not sum_card: # SumCard does not work well with Aggregate class
            c3 = Aggregate([c3, Periodic(365)])
        elif yearly_periodic and sum_card:
            print("Cannot use Periodic Class with SumCard.")

    classes = [c1, c2, c3]

    problem = Problem(signal, classes, use_set=use_ixs)
    problem.decompose(solver=solver, verbose=verbose, eps_rel=1e-6, eps_abs=1e-6)

    s_error =  problem.decomposition[0]
    s_seas = problem.decomposition[1]
    s_hat = problem.decomposition[2]

    if return_all:
        return s_hat, s_seas, s_error, problem

    return s_hat, s_seas


def _osd_tl1_l2d2p365(
        signal,
        use_ixs=None,
        tau=0.75,
        w0=1,
        w1=500,
        yearly_periodic=True,
        return_all=False,
        solver="OSQP",
        verbose=False
):
    """
    Used in:
        solardatatools/algorithms/sunrise_sunset_estimation.py
        solardatatools/clear_day_detection.py
        solardatatools/data_quality.py
        solardatatools/sunrise_sunset.py

    This is a convex problem and the default solver across SDT is OSQP.

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
    c1 = SumQuantile(tau=tau, weight=w0)
    c2 = SumSquare(weight=w1, diff=2)

    if len(signal) > 365 and yearly_periodic:
        c2 = Aggregate([c2, Periodic(365)])

    classes = [c1, c2]

    problem = Problem(signal, classes, use_set=use_ixs)

    problem.decompose(solver=solver, verbose=verbose)
    s_seas = problem.decomposition[1]

    if return_all:
        return s_seas, problem

    return s_seas


def _osd_l1_l1d1_l2d2p365(
    signal,
    use_ixs=None,
    w0=2e-6,  # l1 term, scaled
    w1=40e-6, # l1d1 term, scaled
    w2=6e-3, # seasonal term, scaled
    return_all=False,
    solver=None,
    sum_card=False,
    verbose=False
):
    """
    Used in solardatatools/algorithms/capacity_change.py

    This is a nonconvex problem when invoking QSS and sum_card=True.

    :param signal: A 1d numpy array (must support boolean indexing) containing
    the signal of interest
    :param use_ixs: List of booleans indicating indices to use in signal.
    None is default (uses the entire signal).
    :param w0: Weight on the residual component
    :param w1: The regularization parameter to control the total variation in
    the final output signal
    :param w2: The regularization parameter to control the smoothness of the
    seasonal signal
    :param return_all: Returns all components and the objective value. Used for tests.
    :param solver: Solver to use for the decomposition. QSS and OSQP are supported with
    OSD. MOSEK will trigger CVXPY use.
    :param sum_card: Boolean for using the nonconvex formulation using the cardinality penalty,
    Supported only using OSD with the QSS solver.
    :param verbose: Sets verbosity
    :return: A tuple with three 1d numpy arrays containing the three signal component estimates
    """
    if solver!="QSS":
        sum_card=False

    c1 = SumAbs(weight=w0)

    c2 = SumSquare(weight=w2, diff=2)
    if len(signal) >= 365:
        c2 = Aggregate([c2,
                        AverageEqual(0, period=365),
                        Periodic(365)
                        ])
    else:
        w1 /= 5 # PWC weight needs adjusting when dataset is short

    if sum_card:
        c3 = SumCard(weight=w1, diff=1)
    else:
        c3 = SumAbs(weight=w1, diff=1)

    # Linear term to describe yearly degradation of seasonal component
    c4 =  Aggregate([NoCurvature(),
                     FirstValEqual(0)
                     ])

    classes = [c1, c2, c3, c4]

    problem = Problem(signal, classes, use_set=use_ixs)

    problem.decompose(solver=solver, verbose=verbose, eps_abs=1e-6, eps_rel=1e-6)
    s_seas = problem.decomposition[1]
    s_hat = problem.decomposition[2]
    s_lin = problem.decomposition[3]

    if return_all:
        return s_hat, s_seas, s_lin, problem

    return s_hat, s_seas, s_lin


def _osd_l2_l1d2_constrained(
        signal,
        w0=1,
        w1=5,
        return_all=False,
        solver="OSQP",
        verbose=False
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
    :return: A tuple with returning the signal, the l1d2 component estimate,
     and the weight
    """
    c1 = SumSquare(weight=w0)
    c2 = Aggregate([
        SumAbs(weight=w1, diff=2),
        FirstValEqual(0),
        LastValEqual(1)
    ])

    classes = [c1, c2]

    problem = Problem(signal, classes)
    problem.decompose(solver=solver, verbose=verbose, eps_rel=1e-6, eps_abs=1e-6)

    s_hat = problem.decomposition[1]

    if return_all:
        return s_hat, problem

    return signal, s_hat, w1
