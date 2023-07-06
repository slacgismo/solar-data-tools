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
3) 'l1_l1d1_l2d2p365': like (1) but with an asymmetric residual cost instead
of Gaussian residuals
    - l1: l1-norm
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
    Periodic, Inequality, FirstValEqual, LastValEqual, NoCurvature, NoSlope


def l2_l1d1_l2d2p365(
        signal,
        w0=10,
        w1=50,
        w2=1e5,
        return_all=False,
        yearly_periodic=False,
        solver='QSS',
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
    c2 = Aggregate([SumSquare(weight=w2, diff=2), AverageEqual(0, period=365)])
    if sum_card:
        c3 = SumCard(weight=w1, diff=1)
    else:
        c3 = SumAbs(weight=w1, diff=1)

    if len(signal) > 365:
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

def tl1_l2d2p365(
        signal,
        use_ixs=None,
        tau=0.75,
        w1=500, 
        yearly_periodic=True,
        verbose=False,
        solver='QSS',
        return_all=False
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

    if return_all:
        return s_seas, problem

    return s_seas

def l1_l1d1_l2d2p365(
    signal,
    use_ixs=None,
    w0=2e-6,  # l1 term, scaled
    w1=40e-6, # l1d1 term, scaled
    w2=6e-3, # seasonal term, scaled
    w3=1e-6, # linear term, scaled
    solver=None,
    verbose=False,
    sum_card=False,
    return_all=False
):
    if solver!="QSS":
        sum_card=False

    c1 = SumAbs(weight=w0)
    c2 = Aggregate([SumSquare(weight=w2, diff=2),
                    AverageEqual(0, period=365),
                    Periodic(365)
                    ])

    if sum_card:
        c3 = SumCard(weight=w1, diff=1)
    else:
        c3 = SumAbs(weight=w1, diff=1)

    c4 =  Aggregate([NoCurvature(weight=w3),
                     Inequality(vmin=-0.1, vmax=0.01, diff=1),
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
  
def make_l2_l1d2_constrained(signal,
                            weight=5,
                            solver="QSS",
                            verbose=False,
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

    problem = Problem(signal, classes)
    problem.decompose(solver=solver, verbose=verbose, eps_rel=1e-6, eps_abs=1e-6)

    s_hat = problem.decomposition[1]

    return problem, signal, s_hat, weight