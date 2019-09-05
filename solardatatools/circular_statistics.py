# -*- coding: utf-8 -*-
'''Circular Statistics Module

This module contains functions performing circular statistics. As a general
reference see:

https://www.cambridge.org/core/books/statistical-analysis-of-circular-data/324A46F3941A5CD641ED0B0910B2C33F

'''

import numpy as np

def rayleightest(data, axis=None, weights=None):
    """ Performs the Rayleigh test of uniformity.

    From https://docs.astropy.org/en/stable/_modules/astropy/stats/circstats.html#rayleightest
    (distributed under BSD-3 open-source license)

    This test is  used to identify a non-uniform distribution, i.e. it is
    designed for detecting an unimodal deviation from uniformity. More
    precisely, it assumes the following hypotheses:
    - H0 (null hypothesis): The population is distributed uniformly around the
    circle.
    - H1 (alternative hypothesis): The population is not distributed uniformly
    around the circle.
    Small p-values suggest to reject the null hypothesis.

    Parameters
    ----------
    data : numpy.ndarray or Quantity
        Array of circular (directional) data, which is assumed to be in
        radians whenever ``data`` is ``numpy.ndarray``.
    axis : int, optional
        Axis along which the Rayleigh test will be performed.
    weights : numpy.ndarray, optional
        In case of grouped data, the i-th element of ``weights`` represents a
        weighting factor for each group such that ``np.sum(weights, axis)``
        equals the number of observations.
        See [1]_, remark 1.4, page 22, for detailed explanation.

    Returns
    -------
    p-value : float or dimensionless Quantity
        p-value.


    References
    ----------
    .. [1] S. R. Jammalamadaka, A. SenGupta. "Topics in Circular Statistics".
       Series on Multivariate Analysis, Vol. 5, 2001.
    .. [2] C. Agostinelli, U. Lund. "Circular Statistics from 'Topics in
       Circular Statistics (2001)'". 2015.
       <https://cran.r-project.org/web/packages/CircStats/CircStats.pdf>
    .. [3] M. Chirstman., C. Miller. "Testing a Sample of Directions for
       Uniformity." Lecture Notes, STA 6934/5805. University of Florida, 2007.
    .. [4] D. Wilkie. "Rayleigh Test for Randomness of Circular Data". Applied
       Statistics. 1983.
       <http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.211.4762>
    """
    n = np.size(data, axis=axis)
    Rbar = _length(data, 1, 0.0, axis, weights)
    z = n * Rbar * Rbar

    # see [3] and [4] for the formulae below
    tmp = 1.0
    if(n < 50):
        tmp = 1.0 + (2.0*z - z*z)/(4.0*n) - (24.0*z - 132.0*z**2.0 +
                                             76.0*z**3.0 - 9.0*z**4.0)/(288.0 *
                                                                        n * n)

    p_value = np.exp(-z) * tmp
    return p_value

def _components(data, p=1, phi=0.0, axis=None, weights=None):
    # Utility function for computing the generalized rectangular components
    # of the circular data.
    if weights is None:
        weights = np.ones((1,))
    try:
        weights = np.broadcast_to(weights, data.shape)
    except ValueError:
        raise ValueError('Weights and data have inconsistent shape.')

    C = np.sum(weights * np.cos(p * (data - phi)), axis)/np.sum(weights, axis)
    S = np.sum(weights * np.sin(p * (data - phi)), axis)/np.sum(weights, axis)

    return C, S

def _length(data, p=1, phi=0.0, axis=None, weights=None):
    # Utility function for computing the generalized sample length
    C, S = _components(data, p, phi, axis, weights)
    return np.hypot(S, C)