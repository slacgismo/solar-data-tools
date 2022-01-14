""" Time Shift Algorithm Module

This module contains the algorithm for detecting time shifts in an unlabeled PV
power production data sets. These occur because of the local clock on the data
logging system being changed or by incorrect handling of daylight savings.
The algorithm works as follows:

    - Estimate solar noon on each day from the data
    - Fit a signal demixing model, assuming a seasonal component and a piecewise
      constant component
    - Polish the L1 heuristic used to estimate piecewise constant component
      using iterative reweighting
    - Use piecewise constance component to detect shift points in time and
      correction amounts

"""

import numpy as np
from scipy.stats import mode
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from solardatatools.solar_noon import energy_com, avg_sunrise_sunset
from solardatatools.signal_decompositions import l2_l1d1_l2d2p365


class TimeShift:
    def __init__(self):
        self.metric = None
        self.s1 = None
        self.s2 = None
        self.index_set = None
        self.corrected_data = None
        self.roll_by_index = None
        self.normalized_holdout_error = None
        self.normalized_train_error = None
        self.tv_metric = None
        self.jumps_per_year = None
        self.best_c1 = None
        self.best_ix = None
        self.__recursion_depth = 0

    def run(
        self,
        data,
        use_ixs=None,
        c1=None,
        c2=200.0,
        solar_noon_estimator="com",
        threshold=0.1,
        periodic_detector=False,
        solver=None,
    ):
        if solar_noon_estimator == "com":
            metric = energy_com(data)
        elif solar_noon_estimator == "srss":
            metric = avg_sunrise_sunset(data, threshold=threshold)
        self.metric = metric
        if use_ixs is None:
            use_ixs = ~np.isnan(metric)
        else:
            use_ixs = np.logical_and(use_ixs, ~np.isnan(metric))
        self.use_ixs = use_ixs
        # Optimize c1
        if c1 is None:
            c1s = np.logspace(-1, 2, 11)
            hn, rn, tv_metric, jpy, best_ix = self.optimize_c1(
                metric, c1s, use_ixs, c2, periodic_detector, solver=solver
            )
            if tv_metric[best_ix] >= 0.009:
                # rerun the optimizer with a new random data selection
                hn, rn, tv_metric, jpy, best_ix = self.optimize_c1(
                    metric, c1s, use_ixs, c2, periodic_detector, solver=solver
                )
            # if np.isclose(hn[best_ix], hn[-1]):
            #     best_ix = np.argmax(hn * rn)
            best_c1 = c1s[best_ix]
        else:
            best_c1 = c1
            hn = None
            rn = None
            tv_metric = None
            jpy = None
            c1s = None
            best_ix = None
        s1, s2 = self.estimate_components(
            metric, best_c1, c2, use_ixs, periodic_detector, solver=solver
        )
        # find indices of transition points
        index_set = np.arange(len(s1) - 1)[np.round(np.diff(s1, n=1), 3) != 0]
        # print(len(index_set), len(index_set) / (len(metric) / 365))
        s1, s2 = self.estimate_components(
            metric,
            best_c1,
            c2,
            use_ixs,
            periodic_detector,
            transition_locs=index_set,
            solver=solver,
        )
        jumps_per_year = len(index_set) / (len(metric) / 365)
        cond1 = np.isclose(np.max(s2), 0.5)
        cond2 = c1 is None
        cond3 = self.__recursion_depth < 2
        if cond1 and cond2 and cond3:
            # Unlikely that constraint should be active or that there are more
            # than 5 time shifts per year. Try a different random sampling
            self.__recursion_depth += 1
            self.run(
                data,
                use_ixs=use_ixs,
                c1=c1,
                c2=c2,
                solar_noon_estimator=solar_noon_estimator,
                threshold=threshold,
                periodic_detector=periodic_detector,
                solver=solver,
            )
            return
        # Apply corrections
        roll_by_index = np.round(
            (mode(np.round(s1, 3)).mode[0] - s1) * data.shape[0] / 24, 0
        )
        correction_metric = np.average(np.abs(roll_by_index))
        if correction_metric < 0.01:
            roll_by_index[:] = 0
        self.roll_by_index = roll_by_index
        index_set = np.arange(len(roll_by_index) - 1)[
            np.round(np.diff(roll_by_index, n=1), 3) != 0
        ]
        Dout = self.apply_corrections(data)

        # save results
        self.normalized_holdout_error = hn
        self.normalized_train_error = rn
        self.tv_metric = tv_metric
        self.jumps_per_year = jpy
        self.c1_vals = c1s
        self.best_c1 = best_c1
        self.best_ix = best_ix
        self.s1 = s1
        self.s2 = s2
        self.index_set = index_set
        self.corrected_data = Dout
        self.__recursion_depth = 0

    def optimize_c1(self, metric, c1s, use_ixs, c2, periodic_detector, solver=None):
        # set up train/test split with sklearn
        ixs = np.arange(len(metric))
        ixs = ixs[use_ixs]
        train_ixs, test_ixs = train_test_split(ixs, test_size=0.75)
        train = np.zeros(len(metric), dtype=bool)
        test = np.zeros(len(metric), dtype=bool)
        train[train_ixs] = True
        test[test_ixs] = True
        # initialize results objects
        train_r = np.zeros_like(c1s)
        test_r = np.zeros_like(c1s)
        tv_metric = np.zeros_like(c1s)
        jpy = np.zeros_like(c1s)
        # iterate over possible values of c1 parameter
        for i, v in enumerate(c1s):
            s1, s2 = self.estimate_components(
                metric, v, c2, train, periodic_detector, n_iter=5, solver=solver
            )
            y = metric
            # collect results
            train_r[i] = np.average(np.power((y - s1 - s2)[train], 2))
            test_r[i] = np.average(np.power((y - s1 - s2)[test], 2))
            tv_metric[i] = np.average(np.abs(np.diff(s1, n=1)))
            count_jumps = np.sum(~np.isclose(np.diff(s1), 0, atol=1e-4))
            jumps_per_year = count_jumps / (len(metric) / 365)
            jpy[i] = jumps_per_year

        def zero_one_scale(x):
            return (x - np.min(x)) / (np.max(x) - np.min(x))

        hn = zero_one_scale(test_r)  # holdout error metrix
        rn = zero_one_scale(train_r)
        ixs = np.arange(len(c1s))
        # Detecting more than 5 time shifts per year is extremely uncommon,
        # and is considered non-physical
        slct = jpy <= 5
        best_ix = ixs[slct][np.argmin(hn[slct])]
        return hn, rn, tv_metric, jpy, best_ix

    def estimate_components(
        self,
        metric,
        c1,
        c2,
        use_ixs,
        periodic_detector,
        transition_locs=None,
        n_iter=5,
        solver=None,
    ):
        # Iterative reweighted L1 heuristic
        w = np.ones(len(metric) - 1)
        eps = 0.1
        for i in range(n_iter):
            s1, s2 = l2_l1d1_l2d2p365(
                metric,
                c1=c1,
                c2=c2,
                tv_weights=w,
                use_ixs=use_ixs,
                yearly_periodic=periodic_detector,
                transition_locs=transition_locs,
                seas_max=0.5,
                solver=solver,
            )
            w = 1 / (eps + np.abs(np.diff(s1, n=1)))
        return s1, s2

    def plot_optimization(self, figsize=None):
        if self.best_ix is not None:
            c1s = self.c1_vals
            hn = self.normalized_holdout_error
            rn = self.normalized_train_error
            best_c1 = self.best_c1
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(nrows=4, sharex=True, figsize=figsize)
            ax[0].plot(c1s, hn, marker=".")
            ax[0].axvline(best_c1, ls="--", color="red")
            ax[0].set_title("holdout validation")
            ax[1].plot(c1s, self.jumps_per_year, marker=".")
            ax[1].axvline(best_c1, ls="--", color="red")
            ax[1].set_title("jumps per year")
            ax[2].plot(c1s, rn, marker=".")
            ax[2].axvline(best_c1, ls="--", color="red")
            ax[2].set_title("training residuals")
            ax[3].plot(c1s, self.tv_metric, marker=".")
            ax[3].axvline(best_c1, ls="--", color="red")
            ax[3].set_xscale("log")
            ax[3].set_title("Total variation metric")
            plt.tight_layout()
            return fig

    def apply_corrections(self, data):
        roll_by_index = self.roll_by_index
        Dout = np.copy(data)
        for roll in np.unique(roll_by_index):
            if roll != 0:
                ixs = roll_by_index == roll
                Dout[:, ixs] = np.roll(data, int(roll), axis=0)[:, ixs]
        return Dout

    def invert_corrections(self, data):
        roll_by_index = self.roll_by_index
        Dout = np.copy(data)
        for roll in np.unique(roll_by_index):
            if roll != 0:
                ixs = roll_by_index == roll
                Dout[:, ixs] = np.roll(data, -int(roll), axis=0)[:, ixs]
        return Dout
