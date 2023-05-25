# -*- coding: utf-8 -*-
""" Signal Decompositions Module

This module contains standardized signal decomposition models for use in the
SDT algorithms. The defined signal decompositions are:

1) 'l2_l1d1_l2d2p365': separating a piecewise constant component from a smooth
and seasonal component, with Gaussian noise
    - l2: gaussian noise, sum-of-squares small or l2-norm squared
    - l1d1: piecewise constant heuristic, l1-norm of first order differences
    - l2d2p365: small second order diffs (smooth) and 365-periodic
2) 'tl1_l2d2p365': similar to (2), estimating a smooth, seasonal component with
an asymmetric laplacian noise model, fitting a local quantile instead of a
local average
    - tl1: 'tilted l1-norm,' also known as quantile cost function
    - l2d2p365: small second order diffs (smooth) and 365-periodic
3) 'tl1_l1d1_l2d2p365': like (1) but with an asymmetric residual cost instead
of Gaussian residuals
    - tl1: 'tilted l1-norm,' also known as quantile cost function
    - l1d1: piecewise constant heuristic, l1-norm of first order differences
    - l2d2p365: small second order diffs (smooth) and 365-periodic
4) 'make_l2_l1d2_constrained':
    - l2: gaussian noise, sum-of-squares small or l2-norm squared
    - l1d2: piecewise linear heuristic
    - constrained to have first val at 0 and last val at 1

"""
import sys
import numpy as np

from gfosd import Problem
from gfosd.components import SumAbs, SumSquare, SumCard, SumQuantile, Aggregate, AverageEqual,\
    Periodic, Inequality, FirstValEqual, LastValEqual, NoCurvature


def l2_l1d1_l2d2p365(
        signal,
        w0=10,
        w1=50, # l1d1, c1 in cvxpy version
        w2=1e5, # l2d2, c2 in cvxpy version
        return_all=False,
        yearly_periodic=False,
        solver='MOSEK',
        use_ixs=None,
        sum_card=False,
        verbose=False
):
    """
    This performs total variation filtering with the addition of a seasonal
    baseline fit. This introduces a new signal to the model that is smooth and
    periodic on a yearly time frame. This does a better job of describing real,
    multi-year solar PV power data sets, and therefore does an improved job of
    estimating the discretely changing signal.

    :param signal: A 1d numpy array (must support boolean indexing) containing
    the signal of interest
    :param w1: The regularization parameter to control the total variation in
    the final output signal
    :param w2: The regularization parameter to control the smoothness of the
    seasonal signal
    :return: A 1d numpy array containing the filtered signal
    """

    c1 = SumSquare(weight=w0)
    c2 = Aggregate([SumSquare(weight=w2, diff=2), AverageEqual(0, period=365)])
    if sum_card:
        c3 = SumCard(weight=w1, diff=1)
    else:
        c3 = SumAbs(weight=w1, diff=1) # l1d1 component

    if len(signal) > 365:
        c2 = Aggregate([SumSquare(weight=w2, diff=2), AverageEqual(0, period=365), Periodic(365)])
        if yearly_periodic and not sum_card: # SumCard does not work well with Aggregate class
            c3 = Aggregate([c3, Periodic(365)])
        elif yearly_periodic and sum_card:
            print("Cannot use Periodic Class with SumCard.")

    classes = [c1, c2, c3]

    problem = Problem(signal, classes, use_set=use_ixs)
    problem.decompose(solver=solver, verbose=verbose)

    s_error =  problem.decomposition[0]
    s_seas = problem.decomposition[1]
    s_hat = problem.decomposition[2]

    if return_all:
        return s_hat, s_seas, s_error

    return s_hat, s_seas

def tl1_l2d2p365(
        signal,
        tau=0.75,
        w1=500, # c1 in cvxpy version
        yearly_periodic=True,
        verbose=False,
        solver='MOSEK',
        use_ixs=None
):
    '''
    - tl1: tilted laplacian noise
    - l2d2p365: small second order diffs (smooth) and 365-periodic
    '''

    c1 = SumQuantile(tau=tau, weight=1)
    c2 = SumSquare(weight=w1, diff=2)

    if len(signal) > 365 and yearly_periodic:
        c2 = Aggregate([c2, Periodic(365)])

    classes = [c1, c2]

    problem = Problem(signal, classes, use_set=use_ixs)

    problem.decompose(solver=solver, verbose=verbose)
    s_seas = problem.decomposition[1]

    return s_seas

def tl1_l1d1_l2d2p365( # TODO: switch to l1 since tau passed as 0.5
    signal,
    use_ixs=None,
    tau=0.995, # passed as 0.5
    w0=2,
    w1=1e3, # passed as 15, l1d1 term
    w2=6000,
    w3=1e2, # passed as 300, linear term
    solver=None,
    verbose=False,
    sum_card=False
):
    c1 = SumQuantile(tau=tau, weight=w0)
    c2 = Aggregate([SumSquare(weight=w2, diff=2),
                    AverageEqual(0, period=365)]
                   )

    if sum_card:
        c3 = SumCard(weight=w1, diff=1)
    else:
        c3 = SumAbs(weight=w1, diff=1)

    c4 =  Aggregate([NoCurvature(weight=w3),
                     Inequality(vmin=-0.1, vmax=0.01, diff=1),
                     SumSquare(diff=1),
                     ])

    classes = [c1, c2, c3, c4]

    problem = Problem(signal, classes, use_set=use_ixs)

    problem.decompose(solver=solver, verbose=verbose)
    s_seas = problem.decomposition[1]
    s_hat = problem.decomposition[2]

    return s_hat, s_seas

def make_l2_l1d2_constrained(signal,
                            weight=1e1,
                            solver="MOSEK",
                            use_ixs=None,
                            verbose=False
                             ):
    """
    Used in solardatatools/algorithms/clipping.py
    Added hard-coded constraints on the first and last vals
    """
    c1 = SumSquare(weight=1)
    c2 = Aggregate([
        SumAbs(weight=weight, diff=2),
        FirstValEqual(0),
        LastValEqual(1)
    ])

    classes = [c1, c2]

    problem = Problem(signal, classes, use_set=use_ixs)
    problem.decompose(solver=solver, verbose=verbose)

    s_hat = problem.decomposition[1]

    return s_hat