# -*- coding: utf-8 -*-
""" Signal Decompositions Module using OSD + QSS

This module contains standardized signal decomposition models for use in the
SDT algorithms. The defined signal decompositions are:

1) 'l2_l1d1_l2d2p365': separating a piecewise constant component from a smooth
and seasonal component, with Gaussian noise
    - l2: gaussian noise, sum-of-squares small or l2-norm squared
    - l1d1: piecewise constant heuristic, l1-norm of first order differences
    - l2d2p365: small second order diffs (smooth) and 365-periodic

"""
import sys
import numpy as np
from gfosd import Problem
from gfosd.components import *

def osd_l2_l1d1_l2d2p365(
    signal,
    w1=15e4,
    w2=9e-1,
    solver="QSS",
    verbose=False,
    # residual_weights=None, # make obsolete
    # tv_weights=None, #  does not exist in sd modeling language yet
    # use_ixs=None, # not needed in osd
    yearly_periodic=False,
    # transition_locs=None, # does not exist in osd modeling language yet
    seas_max=None,
):
    """
    This performs total variation filtering with the addition of a seasonal
    baseline fit. This introduces a new signal to the model that is smooth and
    periodic on a yearly time frame. This does a better job of describing real,
    multi-year solar PV power data sets, and therefore does an improved job of
    estimating the discretely changing signal.

    :param signal: A 1d numpy array (must support boolean indexing) containing
    the signal of interest
    :param w1: The regularization parameter to control the smoothness of the
    final output signal
    :param w2: The regularization parameter to control the total variation in the
    seasonal signal
    :return: The three components as 1d numpy arrays
    """
    T = len(signal)

    s_error = SumSquare(weight=1/T)

    s_seas_components = [
        SumSquare(weight=w1/T, diff=2),
        AverageEqual(0, period=365)
    ]
    if yearly_periodic:
        s_seas_components.append(Periodic(365))
    if seas_max:
        s_seas_components.append(Inequality(vmax=seas_max))
    s_seas = Aggregate(s_seas_components)

    s_hat_components = [
        SumAbs(weight=w2 / T, diff=1),
        Inequality(vmax=1, vmin=-1)
    ]
    if yearly_periodic:
        s_hat_components.append(Periodic(365))
    s_hat = Aggregate(s_hat_components)

    components = [s_error, s_seas, s_hat]
    problem = Problem(signal, components)

    problem.decompose(solver=solver, verbose=verbose)
    s_error =  problem.decomposition[0]
    s_seas = problem.decomposition[1]
    s_hat = problem.decomposition[2]

    return s_hat, s_seas, s_error