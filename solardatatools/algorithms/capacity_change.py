""" Capacity Change Algorithm Module

This module the algorithm for detecting capacity changes in an unlabeled PV
power production data sets. The algorithm works as follows:

    - Run daily quantile statistic on cleaned data
    - Fit a signal demixing model, assuming a seasonal component and a piecewise
      constant component
    - Polish the L1 heuristic used to estimate piecewise constant component
      using iterative reweighting
    - Assign daily cluster labels by rounding



"""

import numpy as np
from solardatatools.signal_decompositions import l1_l1d1_l2d2p365
from solardatatools.utilities import segment_diffs, make_pooled_dsig
from sklearn.model_selection import train_test_split


class CapacityChange:
    def __init__(self):
        self.metric = None
        self.s1 = None
        self.s2 = None
        self.s3 = None
        self.labels = None
        self.best_w1 = None
        self.best_ix = None
        self.w1_vals = None
        self.holdout_error = None
        self.test_error = None

    def run(
        self,
        data,
        w1=None,
        filter=None,
        quantile=1.00,
        solver=None,
    ):
        metric = np.nanquantile(data, q=quantile, axis=0)
        # metric /= np.max(metric)
        metric = np.log(metric)
        if filter is None:
            filter = ~np.isnan(metric)
        else:
            filter = np.logical_and(filter, ~np.isnan(metric))
        if w1 is None:
            w1s = np.logspace(-2, 2, 17)
            test_r, train_r, best_ix = self.optimize_weight(
                metric, filter, w1s, solver=solver
            )
            tuned_weight = w1s[best_ix]
        else:
            tuned_weight = w1
            best_ix = None
            test_r = None
            train_r = None
            w1s = None
        # change detector, seasonal term, linear term
        s1, s2, s3 = self.solve_sd(
            metric, filter, tuned_weight, transition_locs=None, solver=solver
        )

        # Identify transition points, and resolve second convex signal decomposition problem
        # find indices of transition points
        seg_diff = segment_diffs(s1)
        no_transitions = len(seg_diff[0]) == 0
        if not no_transitions:
            new_diff = make_pooled_dsig(np.diff(s1), seg_diff)
            transition_locs = np.where(np.abs(new_diff) >= 0.05)[0]
            s1, s2, s3 = self.solve_sd(
                metric,
                filter,
                tuned_weight,
                transition_locs=transition_locs,
                solver=solver,
            )
        # Get capacity assignments (label each day)
        rounded_s1 = custom_round(s1)
        set_labels = list(set(rounded_s1))
        capacity_assignments = [set_labels.index(i) for i in rounded_s1]

        self.metric = metric
        self.s1 = s1  # pwc
        self.s2 = s2  # seasonal
        self.s3 = s3  # linear
        self.labels = capacity_assignments
        self.best_w1 = tuned_weight
        self.best_ix = best_ix
        self.w1_vals = w1s
        self.holdout_error = test_r
        self.train_error = train_r

    def solve_sd(self, metric, filter, w1, transition_locs=None, solver=None):
        s1, s2, s3 = l1_l1d1_l2d2p365(
            metric,
            use_ixs=filter,
            w1=w1,
            transition_locs=transition_locs,
            solver=solver,
            sum_card=False,
        )
        return s1, s2, s3

    def optimize_weight(self, metric, filter, w1s, solver=None):
        ixs = np.arange(len(metric))
        ixs = ixs[filter]
        train_ixs, test_ixs = train_test_split(ixs, train_size=0.85)
        train = np.zeros(len(metric), dtype=bool)
        test = np.zeros(len(metric), dtype=bool)
        train[train_ixs] = True
        test[test_ixs] = True
        # initialize results objects
        train_r = np.zeros_like(w1s)
        test_r = np.zeros_like(w1s)
        # iterate over possible values of w1 parameter
        for i, v in enumerate(w1s):
            s1, s2, s3 = self.solve_sd(metric=metric, filter=train, w1=v, solver=solver)
            y = metric
            # collect results
            train_r[i] = np.average(np.abs((y - s1 - s2 - s3)[train]))
            test_r[i] = np.average(np.abs((y - s1 - s2 - s3)[test]))
        ### Select best weight as the largest weight that gets within a bound, eps, of the lowest holdout error ###
        # find lowest holdout value
        min_test_err = np.min(test_r)
        max_test_err = np.max(test_r)
        # bound is 2% of the test error range, or 2% of the min value, whichever is larger
        bound = max(0.02 * (max_test_err - min_test_err), 0.02 * min_test_err)
        # reduce false positives: holdout error is typically below 1e-3 for all weights when no changes are present
        bound = max(min_test_err + bound, 1e-3)
        cond = test_r <= bound
        ixs = np.arange(len(w1s))
        best_ix = np.max(ixs[cond])
        return test_r, train_r, best_ix


def custom_round(x, base=0.05):
    ndigits = len(str(base).split(".")[-1])
    return np.round(base * np.round(x / base), ndigits)
