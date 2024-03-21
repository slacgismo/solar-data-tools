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
        self.best_w1 = None
        self.best_ix = None
        self.w1_vals = None
        self.baseline = None
        self.correction_estimate = None
        self.periodic_detector = None
        self.__recursion_depth = 0

    def run(
        self,
        data,
        use_ixs=None,
        w1=None,
        w2=200,
        solar_noon_estimator="com",
        threshold=0.005,
        periodic_detector=False,
        solver=None,
        sum_card=False,
        round_shifts_to_hour=True,
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
        # Optimize w1
        if w1 is None:
            # TODO: investigate if two separate ranges are needed
            #  for MOSEK vs QSS and possibly simplify
            if solver == "MOSEK" or solver == "CLARABEL":
                w1s = np.logspace(-1, 2, 11)
            else:
                w1s = np.logspace(0.5, 3.5, 11)
            hn, rn, tv_metric, jpy, best_ix = self.optimize_w1(
                metric,
                w1s,
                use_ixs,
                w2,
                periodic_detector,
                solver=solver,
                sum_card=sum_card,
            )
            if tv_metric[best_ix] >= 0.009:
                # rerun the optimizer with a new random data selection
                hn, rn, tv_metric, jpy, best_ix = self.optimize_w1(
                    metric,
                    w1s,
                    use_ixs,
                    w2,
                    periodic_detector,
                    solver=solver,
                    sum_card=sum_card,
                )
            # if np.isclose(hn[best_ix], hn[-1]):
            #     best_ix = np.argmax(hn * rn)
            best_w1 = w1s[best_ix]
        else:
            best_w1 = w1
            hn = self.normalized_holdout_error
            rn = self.normalized_train_error
            tv_metric = self.tv_metric
            jpy = self.jumps_per_year
            w1s = self.w1_vals
            best_ix = self.best_ix
        s1, s2 = self.estimate_components(
            metric,
            best_w1,
            w2,
            use_ixs,
            periodic_detector,
            solver=solver,
            sum_card=sum_card,
        )
        # find indices of transition points
        index_set = np.arange(len(s1) - 1)[np.round(np.diff(s1, n=1), 3) != 0]
        s1, s2 = self.estimate_components(
            metric,
            best_w1,
            w2,
            use_ixs,
            periodic_detector,
            solver=solver,
            sum_card=sum_card,
            transition_locs=index_set,
        )
        jumps_per_year = len(index_set) / (len(metric) / 365)
        cond1 = jumps_per_year >= 5
        cond2 = w1 is None
        cond3 = self.__recursion_depth < 2
        if cond1 and cond2 and cond3:
            # Unlikely that  there are more than 5 time shifts per year. Try a
            # different random sampling
            self.__recursion_depth += 1
            self.run(
                data,
                use_ixs=use_ixs,
                w1=w1,
                w2=w2,
                solar_noon_estimator=solar_noon_estimator,
                threshold=threshold,
                periodic_detector=periodic_detector,
                solver=solver,
            )
            return
        # Apply corrections
        my_set = set(s1)
        key_func = lambda x: abs(x - 12)
        closest_element = min(my_set, key=key_func)
        if not round_shifts_to_hour:
            roll_by_index = np.round((closest_element - s1) * data.shape[0] / 24, 0)
        else:
            roll_by_index = np.round(
                np.round(closest_element - s1) * data.shape[0] / 24, 0
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
        self.w1_vals = w1s
        self.best_w1 = best_w1
        self.best_ix = best_ix
        self.periodic_detector = periodic_detector
        self.s1 = s1
        self.s2 = s2
        self.index_set = index_set
        self.baseline = closest_element
        self.correction_estimate = roll_by_index * 24 * 60 / data.shape[0]
        self.corrected_data = Dout
        self.__recursion_depth = 0

    def optimize_w1(
        self, metric, w1s, use_ixs, w2, periodic_detector, solver=None, sum_card=False
    ):
        # set up train/test split with sklearn
        ixs = np.arange(len(metric))
        ixs = ixs[use_ixs]
        train_ixs, test_ixs = train_test_split(ixs, train_size=0.85)
        train = np.zeros(len(metric), dtype=bool)
        test = np.zeros(len(metric), dtype=bool)
        train[train_ixs] = True
        test[test_ixs] = True
        # initialize results objects
        train_r = np.zeros_like(w1s)
        test_r = np.zeros_like(w1s)
        tv_metric = np.zeros_like(w1s)
        jpy = np.zeros_like(w1s)
        rms_s2 = np.zeros_like(w1s)
        # iterate over possible values of w1 parameter
        for i, v in enumerate(w1s):
            s1, s2 = self.estimate_components(
                metric,
                v,
                w2,
                train,
                periodic_detector,
                solver=solver,
                sum_card=sum_card,
            )
            y = metric
            # collect results
            train_r[i] = np.average(np.power((y - s1 - s2)[train], 2))
            test_r[i] = np.average(np.power((y - s1 - s2)[test], 2))
            tv_metric[i] = np.average(np.abs(np.diff(s1, n=1)))
            count_jumps = np.sum(~np.isclose(np.diff(s1), 0, atol=1e-4))
            jumps_per_year = count_jumps / (len(metric) / 365)
            jpy[i] = jumps_per_year
            rms_s2[i] = np.sqrt(np.mean(np.square(s2)))
        ### Select best weight as the largest weight that gets within 1% of the lowest holdout error ###
        ixs = np.arange(len(w1s))
        # find lowest holdout value
        min_test_err = np.min(test_r[ixs])
        # add a buffer of 1%
        cond = test_r <= min_test_err * 1.01
        best_ix = np.max(ixs[cond])
        return test_r, train_r, tv_metric, jpy, best_ix

    def estimate_components(
        self,
        metric,
        w1,
        w2,
        use_ixs,
        periodic_detector,
        solver=None,
        sum_card=False,
        transition_locs=None,
    ):

        s1, s2 = l2_l1d1_l2d2p365(
            metric,
            use_ixs=use_ixs,
            w1=w1,
            w2=w2,
            yearly_periodic=periodic_detector,
            solver=solver,
            sum_card=sum_card,
            transition_locs=transition_locs,
        )
        return s1, s2

    def plot_analysis(self, figsize=None):
        if self.metric is not None:
            import matplotlib.pyplot as plt

            fig = plt.figure(figsize=figsize)
            plt.plot(self.metric, linewidth=1, label="metric")
            plt.plot(self.s1, label="shift detector")
            plt.plot(self.s1 + self.s2, ls="--", label="SD model")
            plt.legend()
            plt.xlabel("day number")
            plt.ylabel("solar noon [hours]")
            return fig

    def plot_optimization(self, figsize=None):
        if self.best_ix is not None:
            w1s = self.w1_vals
            hn = self.normalized_holdout_error
            rn = self.normalized_train_error
            best_w1 = self.best_w1
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(nrows=4, sharex=True, figsize=figsize)
            ax[0].plot(w1s, hn, marker=".")
            ax[0].axvline(best_w1, ls="--", color="red")
            ax[0].set_title("holdout validation")
            ax[1].plot(w1s, self.jumps_per_year, marker=".")
            ax[1].set_yscale("log")
            ax[1].axvline(best_w1, ls="--", color="red")
            ax[1].set_title("jumps per year")
            ax[2].plot(w1s, rn, marker=".")
            ax[2].axvline(best_w1, ls="--", color="red")
            ax[2].set_title("training residuals")
            ax[3].plot(w1s, self.tv_metric, marker=".")
            ax[3].axvline(best_w1, ls="--", color="red")
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
