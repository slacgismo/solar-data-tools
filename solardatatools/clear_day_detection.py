# -*- coding: utf-8 -*-
"""Clear Day Detection Module

This module contains functions for detecting clear days in historical PV solar data sets.

"""

import numpy as np
import cvxpy as cvx
from solardatatools.signal_decompositions import tl1_l2d2p365
from solardatatools.utilities import basic_outlier_filter


def filter_for_sparsity(data, c1=6e3, solver="ECOS"):
    nans = np.isnan(data)
    capacity_est = np.nanquantile(data, 0.95)
    data_copy = np.copy(data)
    data_copy[nans] = 0.0 # simplify here

    foo = data_copy > 0.02 * capacity_est  # 2% of 95th perc
    density_signal = np.sum(foo, axis=0) / data.shape[0]
    use_days = np.logical_and(density_signal > 0.2, density_signal < 0.8)

    filtered_signal = tl1_l2d2p365(density_signal, c1=c1, use_ixs=use_days, tau=0.85, solver=solver)
    mask = basic_outlier_filter(density_signal - filtered_signal, outlier_constant=5.0)
    return mask


def find_clear_days(
    data, smoothness_threshold=0.9, energy_threshold=0.8, boolean_out=True, solver=None
):
    """
    This function quickly finds clear days in a PV power data set. The input to this function is a 2D array containing
    standardized time series power data. This will typically be the output from
    `solardatatools.data_transforms.make_2d`. The filter relies on two estimates of daily "clearness": the smoothness
    of each daily signal as measured by the l2-norm of the 2nd order difference, and seasonally-adjusted daily
    the daily energy and a value of 1 will rely entirely on daily smoothness.

    :param D: A 2D numpy array containing a solar power time series signal.
    :param th: A parameter that tunes the filter between relying of daily smoothness and daily energy
    :return: A 1D boolean array, with `True` values corresponding to clear days in the data set
    """
    # Take the norm of the second different of each day's signal. This gives a rough estimate of the smoothness of
    # day in the data set
    tc = np.linalg.norm(data[:-2] - 2 * data[1:-1] + data[2:], ord=1, axis=0)
    # Shift this metric so the median is at zero
    # tc = np.percentile(tc, 50) - tc
    # Normalize such that the maximum value is equal to one
    tc /= np.nanmax(tc)
    tc = 1 - tc
    # Seasonal renormalization: estimate a "baseline smoothness" based on local
    # 90th percentile of smoothness signal. This has the effect of increasing
    # the score of days if there aren't very many smooth days nearby
    # TODO: occasionally fails when CVXPY invokes OSQP instead of MOSEK
    y = tl1_l2d2p365(tc, tau=0.9, c1=2.5e6, yearly_periodic=False, solver=solver)

    tc /= y
    # Take the positive part function, i.e. set the negative values to zero.
    # This is the first metric
    tc = np.clip(tc, 0, None)
    # Calculate the daily energy
    de = np.sum(data, axis=0)
    # Scale by max
    de /= np.nanmax(de)
    # Solve a convex minimization problem to roughly fit the local 90th
    # percentile of the data (quantile regression)
    x = tl1_l2d2p365(de, tau=0.9, c1=204697, yearly_periodic=True, solver=solver)
    # x gives us the local top 90th percentile of daily energy, i.e. the very sunny days. This gives us our
    # seasonal normalization.
    de = np.clip(np.divide(de, x), 0, 1)
    # Take geometric mean
    weights = np.multiply(np.power(tc, 0.5), np.power(de, 0.5))
    # Set values less than 0.6 to be equal to zero
    # weights[weights < 0.6] = 0.
    # Selection rule
    selection = np.logical_and(tc > smoothness_threshold, de > energy_threshold)
    weights[~selection] = 0.0
    # Apply filter for sparsity to catch data errors related to non-zero nighttime data
    msk = filter_for_sparsity(data, solver=None)
    weights = weights * msk.astype(int)
    if boolean_out:
        return weights >= 1e-3
    else:
        return weights
