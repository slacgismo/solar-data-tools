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
from solardatatools.signal_decompositions import l1_pwc_smoothper_trend
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


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
        self.jmp_counts = None
        self.min_jumps = None

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
            slct = np.logical_and(~np.isnan(metric), metric > 0)
            metric_temp[slct] = np.log(metric[slct])
            metric = metric_temp
        if filter is None:
            filter = ~np.isnan(metric)
        else:
            filter = np.logical_and(filter, ~np.isnan(metric))
        if weight is None:
            # initial check if changes are present
            # initial_weight = 50
            # initial_weight = .1
            # s1, s2, s3 = self.solve_sd(
            #     metric=metric, filter=filter, weight=initial_weight, solver=solver
            # )
            # rounded_s1 = custom_round(s1 - s1[0]) + s1[0]
            # set_labels = list(set(rounded_s1))
            # if len(set_labels) == 1:
            #     tuned_weight = initial_weight
            #     best_ix = None
            #     test_r = None
            #     train_r = None
            #     weights = None
            #     jmp_counts = None
            # else:
            # changes are present, optimize the weight with holdout
            # weights = np.logspace(1, 3, 9)
            weights = np.logspace(-0.5, 2.5, 13)
            test_r, train_r, best_ix, jmp_counts, min_jumps = self.optimize_weight(
                metric, filter, weights, solver=solver
            )
            tuned_weight = weights[best_ix]
            s1, s2, s3 = self.solve_sd(metric, filter, tuned_weight, solver=solver)
            num_jumps = np.sum(~np.isclose(np.diff(s1), 0))
            num_years = np.max([1, len(metric) / 365])
            # if num_jumps > num_years * 4:
            #     tuned_weight = weights[-1]
            #     best_ix = len(weights) - 1
            #     s1, s2, s3 = self.solve_sd(metric, filter, tuned_weight, solver=solver)
        else:
            tuned_weight = weight
            best_ix = None
            test_r = None
            train_r = None
            weights = None
            jmp_counts = None
            min_jumps = None
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
        self.jmp_counts = jmp_counts
        self.min_jumps = min_jumps

    def solve_sd(
        self, metric, filter, weight, w3=1, w4=1e1, solver=None, verbose=False
    ):
        s1, s2, s3 = l1_pwc_smoothper_trend(
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
        # train[train_ixs] = True
        # test[test_ixs] = True
        train[filter] = True
        test[filter] = True
        # initialize results objects
        train_r = np.zeros_like(weights)
        test_r = np.zeros_like(weights)
        jmp_counts = np.zeros_like(weights)
        min_jumps = np.zeros_like(weights)
        # iterate over possible values of weight parameter
        for i, v in enumerate(weights):
            s1, s2, s3 = self.solve_sd(
                metric=metric, filter=train, weight=v, solver=solver
            )
            y = metric
            # collect results
            train_r[i] = np.average(np.abs((y - s1 - s2 - s3)[train]))
            test_r[i] = np.average(np.abs((y - s1 - s2 - s3)[test]))
            # beta = np.mean(np.diff(s3))
            # jmpc = np.max(np.convolve(~np.isclose(np.diff(s1), 0), np.ones(365), mode='same'))
            nonzero = ~np.isclose(np.diff(s1), 0, atol=1e-3)
            jmpc = np.sum(nonzero)
            jmp_counts[i] = jmpc
            try:
                min_jumps[i] = np.min(np.abs(np.diff(s1)[nonzero]))
            except ValueError:
                min_jumps[i] = 0
        # Procedure to select weight:
        # We are looking for weights that result in fits that: (1) improve the fit accuracy over not including the pwc term,
        # (2) result in no more than 5 total jumps, and (3) don't produce small changes in the pwc term. We check these
        # conditions below. If no fits qualify, the data does not have capacity changes. If multiple fits quality, we take
        # the smallest qualifying weight. Of the three, the third condition is the most important. Fits with small jumps indicate
        # "overfitting" to the data, i.e. the weight is too small.
        model_fit_improvement = train_r / np.max(train_r)
        cond1 = (
            model_fit_improvement <= 0.98
        )  # 2% minimum improvement over fit with no piecewise constant term
        cond2 = jmp_counts < 5  # no more than 5 jumps in the data set
        cond3 = min_jumps >= 0.075  # the smallest allowable nonzero jump is 0.075
        candidate_ixs = np.all([cond1, cond2, cond3], axis=0)
        if np.sum(candidate_ixs) == 0:
            best_ix = len(weights) - 1
        else:
            best_ix = np.where(candidate_ixs)[0][0]
        return test_r, train_r, best_ix, jmp_counts, min_jumps

    def plot_weight_optimization(self, figsize=(10, 5)):
        """
        A function for plotting plotting the three weight selection criteria

        :param figsize: a 2-tuple of the figure size to plot
        :return: matplotlib figure
        """
        _fig, _ax = plt.subplots(nrows=3, sharex=True, figsize=figsize)
        _ax[0].plot(
            self.weight_vals,
            self.holdout_error / np.max(self.holdout_error),
            marker=".",
        )
        _ax[0].axhline(0.98, color="red", ls="--")
        _ax[0].set_xscale("log")
        _ax[0].set_ylabel("fit improvement")
        _ax[1].plot(
            self.weight_vals,
            self.jmp_counts,
            marker=".",
        )
        _ax[1].axhline(5, color="red", ls="--")
        _ax[1].set_yscale("log")
        _ax[1].set_ylabel("jump count")
        _ax[2].plot(
            self.weight_vals,
            self.min_jumps,
            marker=".",
        )
        _ax[2].axhline(0.075, color="red", ls="--")
        _ax[2].set_ylabel("min nonzero jump")
        _ax[-1].set_xlabel("weight")
        for _i in range(3):
            _ax[_i].axvline(self.best_weight, color="orange", ls=":")
        return _fig


def custom_round(x, base=0.05):
    ndigits = len(str(base).split(".")[-1])
    return np.round(base * np.round(x / base), ndigits)
