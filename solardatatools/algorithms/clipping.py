''' Clipping Module

This module is for analyzing clipping in power data

'''

import numpy as np
import cvxpy as cvx
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

class ClippingDetection():
    def __init__(self):
        self.inverter_clipping = None
        self.num_clip_points = None
        self.num_days = None
        self.clip_stat_1 = None
        self.clip_stat_2 = None
        self.clipped_days = None
        self.threshold = None
        self.cdf_x = None
        self.cdf_y = None
        self.problem = None
        self.y_param = None
        self.weight = None
        self.metric = None
        self.point_masses = None
        self.point_mass_locations = None

    def check_clipping(self, data_matrix, no_error_flag=None, threshold=-0.5,
                       solver='MOSEK', verbose=False, weight=1e1):
        self.num_days = data_matrix.shape[1]
        if no_error_flag is None:
            no_error_flag = np.ones(self.num_days, dtype=bool)
        max_value = np.max(data_matrix)
        daily_max_val = np.max(data_matrix, axis=0)
        # 1st clipping statistic: ratio of the max value on each day to overall max value
        clip_stat_1 = daily_max_val / max_value
        # 2nd clipping statistic: fraction of energy generated each day at or
        # near that day's max value
        with np.errstate(divide='ignore', invalid='ignore'):
            temp = data_matrix / daily_max_val
            temp_msk = temp > 0.995
            temp2 = np.zeros_like(temp)
            temp2[temp_msk] = temp[temp_msk]
            clip_stat_2 = np.sum(temp2, axis=0) / np.sum(temp, axis=0)
        clip_stat_2[np.isnan(clip_stat_2)] = 0
        # Identify which days have clipping
        clipped_days = np.logical_and(
            clip_stat_1 > 0.05,
            clip_stat_2 > 0.1
        )
        clipped_days = np.logical_and(
            no_error_flag,
            clipped_days
        )
        # clipped days must also be near a peak in the distribution of the
        # 1st clipping statistic that shows the characteristic, strongly skewed
        # peak shape
        self.pointmass_detection(
            clip_stat_1, threshold=threshold, solver=solver, verbose=verbose,
            weight=weight
        )
        try:
            if len(self.point_masses) == 0:
                clipped_days[:] = False
            else:
                clipped_days[clipped_days] = np.any(
                    np.array(
                        [np.abs(clip_stat_1[clipped_days] - x0) < .02 for x0 in
                         self.point_masses]), axis=0
                )
        except IndexError:
            self.inverter_clipping = False
            self.num_clip_points = 0
            return
        self.clip_stat_1 = clip_stat_1
        self.clip_stat_2 = clip_stat_2
        self.clipped_days = clipped_days
        if np.sum(clipped_days) > 0.01 * self.num_days:
            self.inverter_clipping = True
            self.num_clip_points = len(self.point_masses)
        else:
            self.inverter_clipping = False
            self.num_clip_points = 0
        return


    def pointmass_detection(self, data, threshold=-0.5, solver='MOSEK',
                            verbose=False, weight=1e1):
        self.threshold = threshold
        x_rs, y_rs = self.calculate_cdf(data)
        self.cdf_x = x_rs
        self.cdf_y = y_rs
        # Fit statistical model to resampled CDF that has sparse 2nd order difference
        if self.problem is None or self.y_param is None:
            self.make_problem(y_rs, weight=weight)
        else:
            self.y_param.value = y_rs
            self.weight.value = weight
        self.problem.solve(solver=solver, verbose=verbose)
        y_hat = self.y_param
        # Look for outliers in the 2nd order difference to identify point masses from clipping
        local_curv = cvx.diff(y_hat, k=2).value
        ref_slope = cvx.diff(y_hat, k=1).value[:-1]
        # metric = local_curv / ref_slope
        metric = np.min([
            local_curv / ref_slope,
            np.concatenate([
                (local_curv[:-1] + local_curv[1:]) / ref_slope[:-1],
                [local_curv[-1] / ref_slope[-1]]
            ]),
            np.concatenate([
                (local_curv[:-2] + local_curv[1:-1] + local_curv[2:]) / ref_slope[:-2],
                [local_curv[-2:] / ref_slope[-2:]]
            ], axis=None)
        ], axis=0)
        # looking for drops of more than 65%
        point_masses = np.concatenate(
            [[False], np.logical_and(metric <= threshold, ref_slope > 3e-4),
             [False]])
        # Catch if the PDF ends in a point mass at the high value
        if np.logical_or(cvx.diff(y_hat, k=1).value[-1] > 1e-3,
                         np.allclose(cvx.diff(y_hat, k=1).value[-1],
                                     np.max(cvx.diff(y_hat, k=1).value))):
            point_masses[-2] = True
        # Reduce clusters of detected points to single points
        pm_reduce = np.zeros_like(point_masses, dtype=np.bool)
        for ix in range(len(point_masses) - 1):
            if ~point_masses[ix] and point_masses[ix + 1]:
                begin_cluster = ix + 1
            elif point_masses[ix] and ~point_masses[ix + 1]:
                end_cluster = ix
                try:
                    ix_select = np.argmax(
                        metric[begin_cluster:end_cluster + 1]
                    )
                except ValueError:
                    pm_reduce[begin_cluster] = True
                else:
                    pm_reduce[begin_cluster + ix_select] = True
        point_masses = pm_reduce
        point_mass_values = x_rs[point_masses]
        self.metric = metric
        self.point_masses = point_masses
        self.point_mass_locations = point_mass_values

    def find_clipped_times(self):
        pass


    def plot_cdf(self, figsize=(8, 6)):
        x_rs = self.cdf_x
        y_rs = self.cdf_y
        y_hat = self.y_hat
        point_masses = self.point_masses
        point_mass_values = self.point_mass_locations
        fig = plt.figure(figsize=figsize)
        plt.plot(x_rs, y_rs, linewidth=1, label='empirical CDF')
        plt.plot(x_rs, y_hat.value, linewidth=3, color='orange', alpha=0.57,
                 label='estimated CDF')
        if len(point_mass_values) > 0:
            plt.scatter(x_rs[point_masses], y_rs[point_masses],
                        color='red', marker='o',
                        label='detected point mass')
        return fig

    def plot_pdf(self, figsize=(8, 6)):
        data = self.clip_stat_1
        x_rs = self.cdf_x
        y_hat = self.y_hat
        point_mass_values = self.point_mass_locations
        fig = plt.figure(figsize=figsize)
        plt.hist(data[data > 0], bins=100, alpha=0.5, label='histogram')
        scale = np.histogram(data[data > 0], bins=100)[0].max() \
                / cvx.diff(y_hat, k=1).value.max()
        plt.plot(x_rs[:-1], scale * cvx.diff(y_hat, k=1).value,
                 color='orange', linewidth=1,
                 label='piecewise constant PDF estimate')
        for count, val in enumerate(point_mass_values):
            if count == 0:
                plt.axvline(val, linewidth=1, linestyle=':',
                            color='green', label='detected point mass')
            else:
                plt.axvline(val, linewidth=1, linestyle=':',
                            color='green')
        return fig

    def plot_diffs(self, figsize=(8, 6)):
        x_rs = self.cdf_x
        metric = self.metric
        threshold = self.threshold
        y_hat = self.y_hat
        point_masses = self.point_masses
        point_mass_values = self.point_mass_locations
        fig, ax = plt.subplots(nrows=2, sharex=True, figsize=figsize)
        y1 = cvx.diff(y_hat, k=1).value
        y2 = metric
        ax[0].plot(x_rs[:-1], y1)
        ax[1].plot(x_rs[1:-1], y2)
        ax[1].axhline(threshold, linewidth=1, color='r', ls=':',
                      label='decision boundary')
        if len(point_mass_values) > 0:
            ax[0].scatter(x_rs[point_masses],
                          y1[point_masses[1:]],
                          color='red', marker='o',
                          label='detected point mass')
            ax[1].scatter(x_rs[point_masses],
                          y2[point_masses[1:-1]],
                          color='red', marker='o',
                          label='detected point mass')
        ax[0].set_title('1st order difference of CDF fit')
        ax[1].set_title('2nd order difference of CDF fit')
        ax[1].legend()
        plt.tight_layout()
        return fig

    def plot_both(self, figsize=(8, 6)):
        data = self.clip_stat_1
        x_rs = self.cdf_x
        y_rs = self.cdf_y
        y_hat = self.y_hat
        point_masses = self.point_masses
        point_mass_values = self.point_mass_locations
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[2, 1])
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(y_rs, x_rs, linewidth=1, label='empirical CDF')
        ax1.plot(y_hat.value, x_rs, linewidth=3, color='orange', alpha=0.57,
                 label='estimated CDF')
        if len(point_mass_values) > 0:
            ax1.scatter(y_rs[point_masses], x_rs[point_masses],
                        color='red', marker='o',
                        label='detected point mass')
        ax1.set_title(
            'Cumulative density function of\nnormalized daily maximum values')
        ax1.set_ylabel('Normalized daily max power')
        ax1.set_xlabel('Cumulative occurrence probability')
        ax1.legend()
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.hist(data[data > 0], bins=100, alpha=0.5, label='histogram',
                 orientation='horizontal')
        scale = np.histogram(data[data > 0], bins=100)[0].max() \
                / cvx.diff(y_hat, k=1).value.max()
        ax2.plot(scale * cvx.diff(y_hat, k=1).value, x_rs[:-1],
                 color='orange', linewidth=1, label='piecewise constant fit')
        for count, val in enumerate(point_mass_values):
            if count == 0:
                plt.axhline(val, linewidth=1, linestyle=':',
                            color='green', label='detected point mass')
            else:
                plt.axhline(val, linewidth=1, linestyle=':',
                            color='green')
        ax2.set_title('Distribution of normalized\ndaily maximum values')
        # ax2.set_ylabel('Normalized daily max power')
        ax2.set_xlabel('Count')
        ax2.legend(loc=(.15, .02))  # -0.4
        return fig

    def calculate_cdf(self, data):
        # Calculate empirical CDF
        x = np.sort(np.copy(data))
        x = x[x > 0]
        x = np.concatenate([[0.], x, [1.]])
        y = np.linspace(0, 1, len(x))
        # Resample the CDF to get an even spacing of points along the x-axis
        f = interp1d(x, y)
        x_rs = np.linspace(0, 1, 5000)
        y_rs = f(x_rs)
        return x_rs, y_rs

    def make_problem(self, y, weight=1e1):
        y_hat = cvx.Variable(len(y))
        y_param = cvx.Parameter(len(y), value=y)
        mu = cvx.Parameter(nonneg=True)
        mu.value = weight
        error = cvx.sum_squares(y_param - y_hat)
        reg = cvx.norm(cvx.diff(y_hat, k=2), p=1)
        objective = cvx.Minimize(error + mu * reg)
        constraints = [
            y_param[0] == y_hat[0],
            y[-1] == y_hat[-1]
        ]
        problem = cvx.Problem(objective, constraints)
        self.problem = problem
        self.y_param = y_param
        self.y_hat = y_hat
        self.weight = mu
