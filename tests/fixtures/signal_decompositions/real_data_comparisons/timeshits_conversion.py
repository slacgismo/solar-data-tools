# for CVXPY implementation
from solardatatools.signal_decompositions import l2_l1d1_l2d2p365 as cvx_sd
from solardatatools.algorithms import TimeShift

# for OSD implementation
import cvxpy as cvx
from scipy.stats import mode
from sklearn.model_selection import train_test_split
from solardatatools.solar_noon import energy_com, avg_sunrise_sunset
from solardatatools.osd_signal_decompositions import l2_l1d1_l2d2p365 as osd_sd

### Define OSD class/function (reference right fn and update arg names, keeping c1 vals the same)

class TimeShiftOSDSimple:
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
        if c1 is None: # unused
            c1s = np.logspace(-1, 2, 11)
            #c1s = np.logspace(0.55, 2, 11)
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
        cond1 = jumps_per_year >= 5
        cond2 = c1 is None
        cond3 = self.__recursion_depth < 2
        if cond1 and cond2 and cond3:
            # Unlikely that  there are more than 5 time shifts per year. Try a
            # different random sampling
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
        train_ixs, test_ixs = train_test_split(ixs, test_size=0.85)
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
        slct = np.logical_and(jpy <= 5, hn <= 0.02)
        # slct = np.logical_and(slct, rn < 0.9)
        best_ix = np.nanmax(ixs[slct])
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
        #for i in range(n_iter): # set this to 1
            #################################################################################
        s1, s2 = osd_sd(
            metric,
            w1=c1,
            w2=c2,
            #tv_weights=w, # do not use this to compare first pass
            use_ixs=use_ixs,
            yearly_periodic=periodic_detector,
            transition_locs=transition_locs,
            #seas_max=None,
            solver=solver,
        )
            #################################################################################
            #w = 1 / (eps + np.abs(np.diff(s1, n=1)))
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
    
    
class TimeShiftCVXSimple:
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
            pass
            # c1s = np.logspace(-1, 2, 11)
            # hn, rn, tv_metric, jpy, best_ix = self.optimize_c1(
            #     metric, c1s, use_ixs, c2, periodic_detector, solver=solver
            # )
            # if tv_metric[best_ix] >= 0.009:
            #     # rerun the optimizer with a new random data selection
            #     hn, rn, tv_metric, jpy, best_ix = self.optimize_c1(
            #         metric, c1s, use_ixs, c2, periodic_detector, solver=solver
            #     )
            # # if np.isclose(hn[best_ix], hn[-1]):
            # #     best_ix = np.argmax(hn * rn)
            # best_c1 = c1s[best_ix]
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
        cond1 = jumps_per_year >= 5
        cond2 = c1 is None
        cond3 = self.__recursion_depth < 2
        if cond1 and cond2 and cond3:
            # Unlikely that  there are more than 5 time shifts per year. Try a
            # different random sampling
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

#     def optimize_c1(self, metric, c1s, use_ixs, c2, periodic_detector, solver=None):
#         # set up train/test split with sklearn
#         ixs = np.arange(len(metric))
#         ixs = ixs[use_ixs]
#         train_ixs, test_ixs = train_test_split(ixs, test_size=0.85)
#         train = np.zeros(len(metric), dtype=bool)
#         test = np.zeros(len(metric), dtype=bool)
#         train[train_ixs] = True
#         test[test_ixs] = True
#         # initialize results objects
#         train_r = np.zeros_like(c1s)
#         test_r = np.zeros_like(c1s)
#         tv_metric = np.zeros_like(c1s)
#         jpy = np.zeros_like(c1s)
#         # iterate over possible values of c1 parameter
#         for i, v in enumerate(c1s):
#             s1, s2 = self.estimate_components(
#                 metric, v, c2, train, periodic_detector, n_iter=5, solver=solver
#             )
#             y = metric
#             # collect results
#             train_r[i] = np.average(np.power((y - s1 - s2)[train], 2))
#             test_r[i] = np.average(np.power((y - s1 - s2)[test], 2))
#             tv_metric[i] = np.average(np.abs(np.diff(s1, n=1)))
#             count_jumps = np.sum(~np.isclose(np.diff(s1), 0, atol=1e-4))
#             jumps_per_year = count_jumps / (len(metric) / 365)
#             jpy[i] = jumps_per_year

#         def zero_one_scale(x):
#             return (x - np.min(x)) / (np.max(x) - np.min(x))

#         hn = zero_one_scale(test_r)  # holdout error metrix
#         rn = zero_one_scale(train_r)
#         ixs = np.arange(len(c1s))
#         # Detecting more than 5 time shifts per year is extremely uncommon,
#         # and is considered non-physical
#         slct = np.logical_and(jpy <= 5, hn <= 0.02)
#         # slct = np.logical_and(slct, rn < 0.9)
#         best_ix = 0 # np.nanmax(ixs[slct])
#         return hn, rn, tv_metric, jpy, best_ix

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
        #for i in range(n_iter): # set this to 1
            #################################################################################
        s1, s2 = cvx_sd(
            metric,
            c1=c1,
            c2=c2,
            tv_weights=w, # do not use this to compare first pass
            use_ixs=use_ixs,
            yearly_periodic=periodic_detector,
            transition_locs=transition_locs,
            #seas_max=None,
            solver=solver,
        )
            #################################################################################
            #w = 1 / (eps + np.abs(np.diff(s1, n=1)))
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
    
class TimeShiftOSD:
    def __init__(self):
        self.atol = 1e-4
        self.sum_card_bool=False
        self.plot_jumps=False
        
        
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
            #c1s = np.logspace(0.3, 2, 15)
            #c1s = [2, 5, 10, 15, 50, 100]
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
        #print("c1 ", best_c1)
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
        cond1 = jumps_per_year >= 5
        cond2 = c1 is None
        cond3 = self.__recursion_depth < 2
        if cond1 and cond2 and cond3:
            # Unlikely that  there are more than 5 time shifts per year. Try a
            # different random sampling
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
        train_ixs, test_ixs = train_test_split(ixs, test_size=0.85)
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
            count_jumps = np.sum(~np.isclose(np.diff(s1), 0, atol=self.atol))
            #######################
            if self.plot_jumps:
                print("count_jumps ", count_jumps)
                idxs = ~np.isclose(np.diff(s1), 0, atol=self.atol)
                fig = plt.figure(figsize=(4,3))
                plt.plot(s1)
                plt.plot(np.arange(1,730,1)[idxs], s1[1:][idxs], marker=".", linewidth=0)
                plt.show()
            ##############
            jumps_per_year = count_jumps / (len(metric) / 365)
            jpy[i] = jumps_per_year

        def zero_one_scale(x):
            return (x - np.min(x)) / (np.max(x) - np.min(x))

        hn = zero_one_scale(test_r)  # holdout error metrix
        rn = zero_one_scale(train_r)
        ixs = np.arange(len(c1s))
        
        # Detecting more than 5 time shifts per year is extremely uncommon,
        # and is considered non-physical
        slct = np.logical_and(jpy <= 5, hn <= 0.02)
        # slct = np.logical_and(slct, rn < 0.9)
        best_ix = np.nanmax(ixs[slct])
        print("OSD c1s ", c1s)
        print("OSD best c1 ", c1s[best_ix])
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
        for i in range(n_iter): # set this to 1
            #################################################################################
            s1, s2 = osd_sd(
                metric,
                w1=c1,
                w2=c2,
                #tv_weights=w, # do not use this to compare first pass
                use_ixs=use_ixs,
                yearly_periodic=periodic_detector,
                transition_locs=transition_locs,
                #seas_max=None,
                solver=solver,
                sum_card=self.sum_card_bool
            )
            #################################################################################
            #w = 1 / (eps + np.abs(np.diff(s1, n=1)))
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
            plt.show()
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
    
class TimeShiftCVX:
    def __init__(self):
        self.atol = 1e-4
        self.iterative_reweight=False
            
            
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
        print("c1 ", best_c1)
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
        cond1 = jumps_per_year >= 5
        cond2 = c1 is None
        cond3 = self.__recursion_depth < 2
        if cond1 and cond2 and cond3:
            # Unlikely that  there are more than 5 time shifts per year. Try a
            # different random sampling
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
        train_ixs, test_ixs = train_test_split(ixs, test_size=0.85)
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
            count_jumps = np.sum(~np.isclose(np.diff(s1), 0, atol=self.atol))
            jumps_per_year = count_jumps / (len(metric) / 365)
            jpy[i] = jumps_per_year

        def zero_one_scale(x):
            return (x - np.min(x)) / (np.max(x) - np.min(x))

        hn = zero_one_scale(test_r)  # holdout error metrix
        rn = zero_one_scale(train_r)
        ixs = np.arange(len(c1s))
        
        self.plot_optimization()
        
        # Detecting more than 5 time shifts per year is extremely uncommon,
        # and is considered non-physical
        slct = np.logical_and(jpy <= 5, hn <= 0.02)
        # slct = np.logical_and(slct, rn < 0.9)
        best_ix = np.nanmax(ixs[slct])
        
        print("CVX c1s ", c1s)
        print("CVX best c1 ", c1s[best_ix])
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
            #################################################################################
            s1, s2 = cvx_sd(
                metric,
                c1=c1,
                c2=c2,
                tv_weights=w, # do not use this to compare first pass
                use_ixs=use_ixs,
                yearly_periodic=periodic_detector,
                transition_locs=transition_locs,
                #seas_max=None,
                solver=solver,
            )
            #################################################################################
            if self.iterative_reweight:
                w = 1 / (eps + np.abs(np.diff(s1, n=1)))
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
