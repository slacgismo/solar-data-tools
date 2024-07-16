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
        metric_temp = np.ones_like(metric) * np.nan
        # metric /= np.max(metric)
        metric[~np.isnan(metric_temp)] = np.log(metric_temp[~np.isnan(metric_temp)])
        if filter is None:
            filter = ~np.isnan(metric)
        else:
            filter = np.logical_and(filter, ~np.isnan(metric))
        if w1 is None:
            w1s = np.logspace(-1, 3, 17)
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
            changes_detected = None
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
        # rounding buckets aligned to first value of component, bin widths of 0.05
        rounded_s1 = custom_round(s1 - s1[0]) + s1[0]
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
        # Precheck: Determine if changes are present. We observe the two conditions indicate a change:
        # 1) there is a significant difference between the best and worst holdout error (otherwise that part of the
        # model is unimportant).
        # 2) Forcing the component to turn off (high weight) results in worse holdout error than including the component
        # with no regularization (low weight).
        holdout_metric = (np.max(test_r) - np.min(test_r)) / np.average(test_r)
        if holdout_metric > 0.2 and test_r[-1] > test_r[0]:
            changes_detected = True
        else:
            changes_detected = False

        if not changes_detected:
            # use largest weight
            best_ix = np.arange(len(w1s))[-1]
        else:
            # Select best weight as the largest weight that doesn't increase the test error by more than the threshold
            # The basic assumption here is that if the test error jumps up sharply when the weight is increased to the
            # point that the piecewise constant term turns off. We make this assumption because we believe a change
            # has been detected.
            try:
                # Try to find the 'large jump' by identifying positive outliers via the interquartile range outlier test
                test_err_diffs = np.diff(test_r)
                upper_quartile = np.percentile(test_err_diffs, 75)
                lower_quartile = np.percentile(test_err_diffs, 25)
                iqr = (upper_quartile - lower_quartile) * 1.5
                holdout_increase_threshold = upper_quartile + iqr
                best_ix = np.where(test_err_diffs > holdout_increase_threshold)[0][0]
            except IndexError:
                # no differences were above the threshold. try another approach.
                # find lowest holdout value
                min_test_err = np.min(test_r)
                max_test_err = np.max(test_r)
                # bound is 2% of the test error range, or 2% of the min value, whichever is larger
                bound = max(0.02 * (max_test_err - min_test_err), 0.02 * min_test_err)
                cond = test_r <= min_test_err + bound
                ixs = np.arange(len(w1s))
                best_ix = np.max(ixs[cond])
        return test_r, train_r, best_ix


def custom_round(x, base=0.05):
    ndigits = len(str(base).split(".")[-1])
    return np.round(base * np.round(x / base), ndigits)
