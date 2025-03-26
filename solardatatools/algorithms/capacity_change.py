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
        self.best_weight = None
        self.best_ix = None
        self.weight_vals = None
        self.holdout_error = None
        self.test_error = None
        self.trend_costs = None

    def run(
        self,
        data,
        metric=None,
        weight=None,
        filter=None,
        quantile=1.00,
        solver=None,
    ):
        if metric is None:
            metric = np.nanquantile(data, q=quantile, axis=0)
            metric_temp = np.ones_like(metric) * np.nan
            # metric /= np.max(metric)
            metric[~np.isnan(metric_temp)] = np.log(metric_temp[~np.isnan(metric_temp)])
        if filter is None:
            filter = ~np.isnan(metric)
        else:
            filter = np.logical_and(filter, ~np.isnan(metric))
        if weight is None:
            # initial check if changes are present
            s1, s2, s3 = self.solve_sd(
                metric=metric, filter=filter, weight=50, solver=solver
            )
            rounded_s1 = custom_round(s1 - s1[0]) + s1[0]
            set_labels = list(set(rounded_s1))
            if len(set_labels) == 1:
                tuned_weight = 50
                best_ix = None
                test_r = None
                train_r = None
                weights = None
                trend_costs = None
            else:
                # changes are present, optimize the weight with holdout
                weights = np.logspace(1, 3, 9)
                test_r, train_r, best_ix, trend_costs = self.optimize_weight(
                    metric, filter, weights, solver=solver
                )
                tuned_weight = weights[best_ix]
                s1, s2, s3 = self.solve_sd(metric, filter, tuned_weight, solver=solver)
        else:
            tuned_weight = weight
            best_ix = None
            test_r = None
            train_r = None
            weights = None
            trend_costs = None
            s1, s2, s3 = self.solve_sd(metric, filter, weight, solver=solver)

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
        self.best_weight = tuned_weight
        self.best_ix = best_ix
        self.weight_vals = weights
        self.holdout_error = test_r
        self.train_error = train_r
        self.trend_costs = trend_costs

    def solve_sd(
        self, metric, filter, weight, w3=1, w4=1e1, solver=None, verbose=False
    ):
        s1, s2, s3 = l1_l1d1_l2d2p365(
            metric,
            use_ixs=filter,
            w2=weight,
            w3=w3,
            w4=w4,
            solver=solver,
            verbose=verbose,
        )
        return s1, s2, s3

    def optimize_weight(self, metric, filter, weights, solver=None):
        ixs = np.arange(len(metric))
        ixs = ixs[filter]
        train_ixs, test_ixs = train_test_split(ixs, train_size=0.7)
        train = np.zeros(len(metric), dtype=bool)
        test = np.zeros(len(metric), dtype=bool)
        train[train_ixs] = True
        test[test_ixs] = True
        # initialize results objects
        train_r = np.zeros_like(weights)
        test_r = np.zeros_like(weights)
        trend_costs = np.zeros_like(weights)
        # iterate over possible values of weight parameter
        for i, v in enumerate(weights):
            s1, s2, s3 = self.solve_sd(
                metric=metric, filter=train, weight=v, solver=solver
            )
            y = metric
            # collect results
            train_r[i] = np.average(np.abs((y - s1 - s2 - s3)[train]))
            test_r[i] = np.average(np.abs((y - s1 - s2 - s3)[test]))
            beta = np.mean(np.diff(s3))
            trend_cost = 1e1 * len(metric) * beta**2
            trend_costs[i] = trend_cost
        # Procedure to select weight:
        # When capacity changes are present, you get lower holdout error by allowing jumps, so the largest errors are
        # with the highest weights. We max/min scale the data, and then select the largest weight that reduces the holdout
        # error by at least half. In short we don't minimize the holdout error, but select the largest weight that gives
        # "significant improvement" over the model with no jumps. Reducing the weight further tends to introduce unwanted
        # changes in the piecwise constant term, and it only makes marginal improvements in the holdout error (i.e the slope
        # error versus weight curve is flatter at lower weights and steeper at higher weights before "turning off" the
        # component).
        max_r = np.max(test_r)
        min_r = np.min(test_r)
        scaled_r = (test_r - min_r) / (max_r - min_r)
        best_ix = np.where(scaled_r <= 0.5)[0][-1]
        return test_r, train_r, best_ix, trend_costs


def custom_round(x, base=0.05):
    ndigits = len(str(base).split(".")[-1])
    return np.round(base * np.round(x / base), ndigits)
