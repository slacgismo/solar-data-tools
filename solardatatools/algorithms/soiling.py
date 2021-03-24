# -*- coding: utf-8 -*-
''' Soiling Module

This module is for analyzing soiling trends in performance index (PI) data/

'''

import numpy as np
import cvxpy as cvx


def soiling_seperation(observed, iterations=5, weights=None, index_set=None,
                       tau=0.85, c1=100, c2=2):
    if weights is None:
        weights =  np.ones_like(observed)
    if index_set is None:
        index_set = ~np.isnan(observed)
    else:
        index_set = np.logical_and(index_set, ~np.isnan(observed))
    zero_set = np.zeros(len(observed) - 1, dtype=np.bool)
    eps = .01
    n = len(observed)
    s1 = cvx.Variable(n)
    s2 = cvx.Variable(max(n, 367))
    s3 = cvx.Variable(n)
    w = cvx.Parameter(n - 2, nonneg=True)
    w.value = np.ones(len(observed) - 2)
    for i in range(iterations):
        # cvx.norm(cvx.multiply(s3, weights), p=2) \
        cost = cvx.sum(tau * cvx.pos(s3) +(1 - tau) * cvx.neg(s3)) \
               + c1 * cvx.norm(cvx.diff(s2[:n], k=2), p=2) \
               + c2 * cvx.norm(cvx.multiply(w, cvx.diff(s1, k=2)), p=1)
        objective = cvx.Minimize(cost)
        constraints = [
            observed[index_set] == s1[index_set] + s2[:n][index_set] + s3[index_set],
            s2[365:] - s2[:-365] == 0,
            cvx.sum(s2[:365]) == 0
            # s1 <= 1
        ]
        if np.sum(zero_set) > 0:
            constraints.append(cvx.diff(s1, k=1)[zero_set] == 0)
        if n < 0.75 * 365:
            constraints.append(s2 == 0)
        problem = cvx.Problem(objective, constraints)
        problem.solve(solver='MOSEK')
        w.value = 1 / (eps + 1e2* np.abs(cvx.diff(s1, k=2).value))   # Reweight the L1 penalty
        zero_set = np.abs(cvx.diff(s1, k=1).value) <= 5e-5     # Make nearly flat regions exactly flat (sparse 1st diff)
    return s1.value, s2.value[:n], s3.value