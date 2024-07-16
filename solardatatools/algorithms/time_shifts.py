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
from solardatatools.utilities import segment_diffs, make_pooled_dsig


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
        w2=1e-3,
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
        # print('in w1 optimization loop. solver is:', solver)
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
            # if tv_metric[best_ix] >= 0.009:
            #     # rerun the optimizer with a new random data selection
            #     hn, rn, tv_metric, jpy, best_ix = self.optimize_w1(
            #         metric,
            #         w1s,
            #         use_ixs,
            #         w2,
            #         periodic_detector,
            #         solver=solver,
            #         sum_card=sum_card,
            #     )
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
        # Solve first convex signal decomposition problem
        s1, s2 = self.estimate_components(
            metric,
            best_w1,
            w2,
            use_ixs,
            periodic_detector,
            solver=solver,
            sum_card=sum_card,
        )
        # Identify transition points, and resolve second convex signal decomposition problem
        # find indices of transition points
        seg_diff = segment_diffs(s1)
        no_transitions = len(seg_diff[0]) == 0
        if not no_transitions:
            new_diff = make_pooled_dsig(np.diff(s1), seg_diff)
            transition_locs = np.where(np.abs(new_diff) >= 0.05)[0]
            s1, s2 = self.estimate_components(
                metric,
                best_w1,
                w2,
                use_ixs,
                periodic_detector,
                solver=solver,
                sum_card=sum_card,
                transition_locs=transition_locs,
            )
            jumps_per_year = len(transition_locs) / (len(metric) / 365)
            # cond1 = jumps_per_year >= 5
            # cond2 = w1 is None
            # cond3 = self.__recursion_depth < 2
            # if cond1 and cond2 and cond3:
            #     # Unlikely that  there are more than 5 time shifts per year. Try a
            #     # different random sampling
            #     self.__recursion_depth += 1
            #     self.run(
            #         data,
            #         use_ixs=use_ixs,
            #         w1=w1,
            #         w2=w2,
            #         solar_noon_estimator=solar_noon_estimator,
            #         threshold=threshold,
            #         periodic_detector=periodic_detector,
            #         transition_locs=transition_locs,
            #         solver=solver,
            #     )
            #     return
        # Apply corrections
        # Set baseline for corrections. We use the clock at the beginning of the records as the baseline, unless we have
        # reason to think the values at the start are bad, like if there's a large deviation from noon or the first
        # changepoint is very close to the start, then we find the times closest to 12-noon.
        closest_element = s1[0]
        if (
            np.abs(closest_element - 12) > 0.9
            or np.sum(s1 == closest_element) / len(s1) < 0.02
        ):
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
            best_ix = len(w1s) - 1
        else:
            try:
                # Try to find the 'large jump' by identifying positive outliers via the interquartile range outlier test
                test_err_diffs = np.diff(test_r)
                upper_quartile = np.percentile(test_err_diffs, 75)
                lower_quartile = np.percentile(test_err_diffs, 25)
                iqr = (upper_quartile - lower_quartile) * 1.5
                holdout_increase_threshold = upper_quartile + iqr
                best_ix = np.where(test_err_diffs > holdout_increase_threshold)[0][0]
            except IndexError:
                # Select best weight as the largest weight that gets within a bound, eps, of the lowest holdout error
                # find lowest holdout value
                min_test_err = np.min(test_r)
                max_test_err = np.max(test_r)
                # bound is 2% of the test error range, or 2% of the min value, whichever is larger
                bound = max(0.02 * (max_test_err - min_test_err), 0.02 * min_test_err)
                # reduce false positives: holdout error is typically below 5e-3 for all weights when no shifts
                bound = max(min_test_err + bound, 5e-3)
                cond = test_r <= bound
                ixs = np.arange(len(w1s))
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

    def plot_optimization(self, figsize=(8, 8)):
        if self.best_ix is not None:
            w1s = self.w1_vals
            hn = self.normalized_holdout_error
            rn = self.normalized_train_error
            best_w1 = self.best_w1
            # find lowest holdout value
            min_test_err = np.min(hn)
            max_test_err = np.max(hn)
            # bound is 2% of the test error range
            bound = max(0.02 * (max_test_err - min_test_err), 0.02 * min_test_err)
            bound = min_test_err + bound
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(nrows=4, sharex=True, figsize=figsize)
            ax[0].plot(w1s, hn, marker=".")
            ax[0].axvline(best_w1, ls="--", color="red")
            # ax[0].axhline(min_test_err, linewidth=1, color="orange")
            # ax[0].axhline(bound, linewidth=1, color="green")
            # ax[0].set_yscale("log")
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
