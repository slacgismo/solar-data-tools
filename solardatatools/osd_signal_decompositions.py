# -*- coding: utf-8 -*-
""" Signal Decompositions Module

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

"""
import sys
import numpy as np

from gfosd import Problem
from gfosd.components import SumAbs, SumSquare, SumCard, SumQuantile, Aggregate, AverageEqual, Periodic


def l2_l1d1_l2d2p365(
        signal,
        w0=10, # error
        w1=100, # l1d1, c1 in cvxpy version
        w2=1, # l2d2, c2 in cvxpy version
        return_all=False,
        yearly_periodic=False,
        solver='MOSEK',
        #tv_weights=None,
        transition_locs=None,
        use_ixs=None,
        sum_card=False
):

    c1 = SumSquare(weight=w0)
    c2 = Aggregate([SumSquare(weight=w2, diff=2), AverageEqual(0, period=365)])
    if sum_card:
        c3 = SumCard(weight=w1, diff=1)
    else:
        c3 = SumAbs(weight=w1, diff=1) # l1d1 component

    if len(signal) > 365:
        c2 = Aggregate([SumSquare(weight=w2, diff=2), AverageEqual(0, period=365), Periodic(365)])
        if yearly_periodic:
            c3 = Aggregate([c3, Periodic(365)])

    classes = [c1, c2, c3]

    problem = Problem(signal, classes, use_set=use_ixs)
    problem.decompose(solver=solver)

    s_error =  problem.decomposition[0]
    s_seas = problem.decomposition[1]
    s_hat = problem.decomposition[2]

    if return_all:
        return s_hat, s_seas, s_error

    return s_hat, s_seas

def l1_l2d2p365(
        signal,
        solver,
        w1=1,
        w2=1e5, # c1 in cvxpy version
        yearly_periodic=True,
        verbose=False,
        use_ixs=None
):
    '''
    - l1: laplacian noise, sum-of-absolute values or l1-norm
    - l2d2p365: small second order diffs (smooth) and 365-periodic
    '''

    c1 = SumAbs(w1)
    c2 = SumSquare(weight=w2, diff=2)

    if len(signal) > 365 and yearly_periodic:
        c2 = Aggregate([c2, Periodic(365)])

    classes = [c1, c2]

    problem = Problem(signal, classes, use_set=use_ixs)

    problem.decompose(solver=solver)
    s_seas = problem.decomposition[1]

    return s_seas

def tl1_l2d2p365(
        signal,
        tau=0.75,
        w1=1,
        w2=1e5, # c1 in cvxpy version
        yearly_periodic=True,
        verbose=False,
        solver='MOSEK',
        use_ixs=None
):
    '''
    - tl1: tilted laplacian noise
    - l2d2p365: small second order diffs (smooth) and 365-periodic
    '''

    c1 = SumQuantile(tau=tau, weight=w1)
    c2 = SumSquare(weight=w2, diff=2)

    if len(signal) > 365 and yearly_periodic:
        c2 = Aggregate([c2, Periodic(365)])

    classes = [c1, c2]

    problem = Problem(signal, classes, use_set=use_ixs)

    problem.decompose(solver=solver)
    s_seas = problem.decomposition[1]

    return s_seas