# -*- coding: utf-8 -*-
''' Data Handler Module

This module contains a class for managing a data processing pipeline

'''
from time import time
import numpy as np
from scipy.stats import mode, skew
from scipy.interpolate import interp1d
from sklearn.cluster import DBSCAN
import cvxpy as cvx
import matplotlib.pyplot as plt
from solardatatools.time_axis_manipulation import make_time_series,\
    standardize_time_axis, fix_time_shifts
from solardatatools.matrix_embedding import make_2d
from solardatatools.data_quality import daily_missing_data_advanced,\
    daily_missing_data_simple, dataset_quality_score
from solardatatools.data_filling import zero_nighttime, interp_missing
from solardatatools.clear_day_detection import find_clear_days
from solardatatools.plotting import plot_2d

class DataHandler():
    def __init__(self, data_frame=None, raw_data_matrix=None,
                 convert_to_ts=False):
        self.data_frame = data_frame
        self.raw_data_matrix = raw_data_matrix
        if self.raw_data_matrix is not None:
            self.num_days = self.raw_data_matrix.shape[1]
            self.data_sampling = int(24 * 60 / self.raw_data_matrix.shape[0])
        else:
            self.num_days = None
            self.data_sampling = None
        self.filled_data_matrix = None
        self.keys = None
        self.use_column = None
        self.capacity_estimate = None
        # Scores for the entire data set
        self.data_quality_score = None      # Fraction of days without data acquisition errors
        self.data_clearness_score = None    # Fraction of days that are approximately clear/sunny
        # Flags for the entire data set
        self.inverter_clipping = None       # True if there is inverter clipping, false otherwise
        self.num_clip_points = None         # If clipping, the number of clipping set points
        self.capacity_changes = None        # True if the apparent capacity seems to change over the data set
        self.normal_quality_scores = None
        # Daily scores (floats) and flags (booleans)
        self.daily_scores = DailyScores()
        self.daily_flags = DailyFlags()
        # Useful daily signals defined by the data set
        self.daily_signals = DailySignals()
        if np.alltrue([data_frame is not None, convert_to_ts]):
            df_ts, keys = make_time_series(self.data_frame)
            self.data_frame = df_ts
            self.keys = keys
        # Private attributes
        self.__time_axis_standardized = False
        self.__density_lower_threshold = None
        self.__density_upper_threshold = None
        self.__linearity_threshold = None

    def run_pipeline(self, use_col=None, zero_night=True, interp_day=True,
                     fix_shifts=True, density_lower_threshold=0.6,
                     density_upper_threshold=1.05, linearity_threshold=0.1,
                     clear_tune_param=0.1, verbose=True, start_day_ix=None,
                     end_day_ix=None):
        t0 = time()
        if self.data_frame is not None:
            self.make_data_matrix(use_col, start_day_ix=start_day_ix,
                                  end_day_ix=end_day_ix)
        t1 = time()
        self.make_filled_data_matrix(zero_night=zero_night, interp_day=interp_day)
        t2 = time()
        self.capacity_estimate = np.quantile(self.filled_data_matrix, 0.95)
        if fix_shifts:
            self.auto_fix_time_shifts()
        t3 = time()
        self.get_daily_scores(threshold=0.2)
        t4 = time()
        self.get_daily_flags(density_lower_threshold=density_lower_threshold,
                             density_upper_threshold=density_upper_threshold,
                             linearity_threshold=linearity_threshold)
        t5 = time()
        self.detect_clear_days(th=clear_tune_param)
        t6 = time()
        self.clipping_check()
        t7 = time()
        self.score_data_set()
        t8 = time()
        if verbose:
            out = 'total time: {:.2f} seconds\n'
            out += 'form matrix: {:.2f}, '
            out += 'fill matrix: {:.2f}, '
            out += 'fix time shifts: {:.2f}, '
            out += 'daily scores: {:.2f}, \n'
            out += 'daily flags: {:.2f}, '
            out += 'clear detect: {:.2f}, '
            out += 'clipping check: {:.2f}, '
            out += 'data scoring: {:.2f}'
            print(out.format(t8-t0, t1-t0, t2-t1, t3-t2, t4-t3, t5-t4, t6-t5, t7-t6, t8-t7))
        return

    def report(self):
        try:
            l1 = 'Length:               {} days\n'.format(self.num_days)
            l2 = 'Data sampling:        {} minute\n'.format(self.data_sampling)
            l3 = 'Data quality score:   {:.1f}%\n'.format(self.data_quality_score * 100)
            l4 = 'Data clearness score: {:.1f}%\n'.format(self.data_clearness_score * 100)
            l5 = 'Inverter clipping:    {}'.format(self.inverter_clipping)
            p_out = l1 + l2 + l3 + l4 + l5
            print(p_out)
            if self.num_clip_points > 1:
                print('WARNING: {} clipping set points detected!'.format(
                    self.num_clip_points
                ))
            if not self.normal_quality_scores:
                print('WARNING: Abnormal clustering of data quality scores!')
            return
        except TypeError:
            print('Please run the pipeline first!')
            return

    def make_data_matrix(self, use_col=None, start_day_ix=None, end_day_ix=None):

        if not self.__time_axis_standardized:
            df = standardize_time_axis(self.data_frame)
            self.data_frame = df
            self.__time_axis_standardized = True
        else:
            df = self.data_frame
        if use_col is None:
            use_col = df.columns[0]
        self.raw_data_matrix = make_2d(df, key=use_col)
        self.raw_data_matrix = self.raw_data_matrix[:, start_day_ix:end_day_ix]
        self.num_days = self.raw_data_matrix.shape[1]
        self.data_sampling = int(24 * 60 / self.raw_data_matrix.shape[0])
        self.use_column = use_col
        return

    def make_filled_data_matrix(self, zero_night=True, interp_day=True):
        self.filled_data_matrix = np.copy(self.raw_data_matrix)
        if zero_night:
            self.filled_data_matrix = zero_nighttime(self.raw_data_matrix)
        if interp_day:
            self.filled_data_matrix = interp_missing(self.filled_data_matrix)
        else:
            msk = np.isnan(self.filled_data_matrix)
            self.filled_data_matrix[msk] = 0
        self.daily_signals.energy = np.sum(self.filled_data_matrix, axis=0) *\
                                   24 / self.filled_data_matrix.shape[1]
        return

    def get_daily_scores(self, threshold=0.2):
        self.get_density_scores(threshold=threshold)
        self.get_linearity_scores()
        return

    def get_daily_flags(self, density_lower_threshold=0.6,
                        density_upper_threshold=1.05, linearity_threshold=0.1):
        self.daily_flags.density = np.logical_and(
            self.daily_scores.density > density_lower_threshold,
            self.daily_scores.density < density_upper_threshold
        )
        self.daily_flags.linearity = self.daily_scores.linearity < linearity_threshold
        self.daily_flags.flag_no_errors()

        scores = np.c_[self.daily_scores.density, self.daily_scores.linearity]
        db = DBSCAN(eps=.03,
                    min_samples=max(0.01 * scores.shape[0], 3)).fit(scores)
        # Count the number of days that cluster to the main group but fall
        # outside the decision boundaries
        day_count = np.logical_or(
            self.daily_scores.linearity[db.labels_ == 0] > linearity_threshold,
            np.logical_or(
                self.daily_scores.density[db.labels_ == 0] < density_lower_threshold,
                self.daily_scores.density[db.labels_ == 0] > density_upper_threshold
            )
        )
        self.normal_quality_scores = np.sum(day_count) <= max(5e-3 * self.num_days, 1)
        self.__density_lower_threshold = 0.6
        self.__density_upper_threshold = 1.05
        self.__linearity_threshold = 0.1
        self.daily_scores.quality_clustering = db.labels_

    def get_density_scores(self, threshold=0.2):
        if self.raw_data_matrix is None:
            print('Generate a raw data matrix first.')
            return
        self.daily_scores.density, self.daily_signals.density, self.daily_signals.seasonal_density_fit\
            = daily_missing_data_advanced(
            self.raw_data_matrix,
            threshold=threshold,
            return_density_signal=True,
            return_fit=True
        )
        return

    def get_linearity_scores(self):
        if self.capacity_estimate is None:
            self.capacity_estimate = np.quantile(self.filled_data_matrix, 0.95)
        if self.daily_signals.seasonal_density_fit is None:
            print('Run the density check first')
            return
        temp_mat = np.copy(self.filled_data_matrix)
        temp_mat[temp_mat < 0.005 * self.capacity_estimate] = np.nan
        difference_mat = np.round(temp_mat[1:] - temp_mat[:-1], 4)
        modes, counts = mode(difference_mat, axis=0, nan_policy='omit')
        n = self.filled_data_matrix.shape[0] - 1
        self.daily_scores.linearity = counts.data.squeeze() / (n * self.daily_signals.seasonal_density_fit)
        return

    def score_data_set(self):
        num_days = self.raw_data_matrix.shape[1]
        self.data_quality_score = np.sum(self.daily_flags.no_errors) / num_days
        self.data_clearness_score = np.sum(self.daily_flags.clear) / num_days
        return

    def clipping_check(self):
        max_value = np.max(self.filled_data_matrix)
        daily_max_val = np.max(self.filled_data_matrix, axis=0)
        # 1st clipping statistic: ratio of the max value on each day to overall max value
        clip_stat_1 = daily_max_val / max_value
        # 2nd clipping statistic: fraction of time each day spent near that day's max value
        with np.errstate(divide='ignore', invalid='ignore'):
            temp = self.filled_data_matrix / daily_max_val
            clip_stat_2 = np.sum(temp > 0.995, axis=0) / np.sum(temp > 0.005, axis=0)
        clip_stat_2[np.isnan(clip_stat_2)] = 0
        # Identify which days have clipping
        clipped_days = np.logical_and(
            clip_stat_1 > 0.05,
            clip_stat_2 > 0.1
        )
        clipped_days = np.logical_and(
            self.daily_flags.no_errors,
            clipped_days
        )
        # clipped days must also be near a peak in the distribution of the
        # 1st clipping statistic that shows the characteristic, strongly skewed
        # peak shape
        point_masses = self.__analyze_distribution(clip_stat_1[self.daily_flags.no_errors])
        if len(point_masses) == 0:
            clipped_days[:] = False
        else:
            clipped_days[clipped_days] = np.any(
                np.array([np.abs(clip_stat_1[clipped_days] - x0) < .02 for x0 in
                          point_masses]), axis=0
            )
        self.daily_scores.clipping_1 = clip_stat_1
        self.daily_scores.clipping_2 = clip_stat_2
        self.daily_flags.inverter_clipped = clipped_days
        if np.sum(clipped_days) > 0:
            self.inverter_clipping = True
            self.num_clip_points = len(point_masses)
        else:
            self.inverter_clipping = False
            self.num_clip_points = 0
        return


    def auto_fix_time_shifts(self, c1=5., c2=500.):
        self.filled_data_matrix = fix_time_shifts(self.filled_data_matrix,
                                                  solar_noon_estimator='srsn',
                                                  c1=c1, c2=c2)

    def detect_clear_days(self, th=0.1):
        if self.filled_data_matrix is None:
            print('Generate a filled data matrix first.')
            return
        clear_days = find_clear_days(self.filled_data_matrix, th=th)
        self.daily_flags.flag_clear_cloudy(clear_days)
        return

    def plot_heatmap(self, matrix='raw', flag=None, figsize=(12, 6)):
        if matrix == 'raw':
            mat = self.raw_data_matrix
        elif matrix == 'filled':
            mat = self.filled_data_matrix
        else:
            return
        if flag is None:
            return plot_2d(mat, figsize=figsize)
        elif flag == 'good':
            fig = plot_2d(mat, figsize=figsize,
                           clear_days=self.daily_flags.no_errors)
            plt.title('Measured power, good days flagged')
            return fig
        elif flag == 'bad':
            fig = plot_2d(mat, figsize=figsize,
                          clear_days=~self.daily_flags.no_errors)
            plt.title('Measured power, bad days flagged')
            return fig
        elif flag in ['clear', 'sunny']:
            fig = plot_2d(mat, figsize=figsize,
                          clear_days=self.daily_flags.clear)
            plt.title('Measured power, clear days flagged')
            return fig
        elif flag == 'cloudy':
            fig = plot_2d(mat, figsize=figsize,
                          clear_days=self.daily_flags.cloudy)
            plt.title('Measured power, cloudy days flagged')
            return fig

    def plot_daily_signals(self, boolean_index=None, day_start=0, num_days=5,
                           filled=True, ravel=True, figsize=(12, 6)):
        if boolean_index is None:
            boolean_index = np.s_[:]
        i = day_start
        j = day_start + num_days
        slct = np.s_[np.arange(self.num_days)[boolean_index][i:j]]
        if filled:
            plot_data = self.filled_data_matrix[:, slct]
        else:
            plot_data = self.raw_data_matrix[:, slct]
        if ravel:
            plot_data = plot_data.ravel(order='F')
        fig = plt.figure(figsize=figsize)
        plt.plot(plot_data, linewidth=1)
        return fig

    def plot_density_signal(self, flag=None, show_fit=False, figsize=(8, 6)):
        if self.daily_signals.density is None:
            return
        fig = plt.figure(figsize=figsize)
        plt.plot(self.daily_signals.density, linewidth=1)
        xs = np.arange(len(self.daily_signals.density))
        title = 'Daily signal density'
        if flag == 'density':
            plt.scatter(xs[~self.daily_flags.density],
                        self.daily_signals.density[~self.daily_flags.density],
                        color='red')
            title += ', density outlier days flagged'
        if flag == 'good':
            plt.scatter(xs[self.daily_flags.no_errors],
                        self.daily_signals.density[self.daily_flags.no_errors],
                        color='red')
            title += ', days that failed density test flagged'
        elif flag == 'bad':
            plt.scatter(xs[~self.daily_flags.no_errors],
                        self.daily_signals.density[~self.daily_flags.no_errors],
                        color='red')
            title += ', bad days flagged'
        elif flag in ['clear', 'sunny']:
            plt.scatter(xs[self.daily_flags.clear],
                        self.daily_signals.density[self.daily_flags.clear],
                        color='red')
            title += ', clear days flagged'
        elif flag == 'cloudy':
            plt.scatter(xs[self.daily_flags.cloudy],
                        self.daily_signals.density[self.daily_flags.cloudy],
                        color='red')
            title += ', cloudy days flagged'
        if np.logical_and(show_fit,
                          self.daily_signals.seasonal_density_fit is not None):
            plt.plot(self.daily_signals.seasonal_density_fit, color='orange')
            plt.plot(0.6 * self.daily_signals.seasonal_density_fit,
                     color='green', linewidth=1,
                     ls='--')
            plt.plot(1.05 * self.daily_signals.seasonal_density_fit,
                     color='green', linewidth=1,
                     ls='--')
        plt.title(title)
        return fig

    def plot_data_quality_scatter(self, figsize=(6,5)):
        fig = plt.figure(figsize=figsize)
        labels = self.daily_scores.quality_clustering
        for lb in set(labels):
            plt.scatter(self.daily_scores.density[labels == lb],
                        self.daily_scores.linearity[labels == lb],
                        marker='.', label=lb)
        plt.xlabel('density score')
        plt.ylabel('linearity score')
        plt.axhline(self.__linearity_threshold, linewidth=1, color='red',
                    ls=':', label='decision boundary')
        plt.axvline(self.__density_upper_threshold, linewidth=1, color='red',
                    ls=':')
        plt.axvline(self.__density_lower_threshold, linewidth=1, color='red',
                    ls=':')
        plt.legend()
        return fig

    def plot_daily_energy(self, flag=None, figsize=(8, 6)):
        if self.filled_data_matrix is None:
            return
        fig = plt.figure(figsize=figsize)
        energy = self.daily_signals.energy
        plt.plot(energy, linewidth=1)
        xs = np.arange(len(energy))
        title = 'Daily energy production'
        if flag == 'good':
            plt.scatter(xs[self.daily_flags.no_errors],
                        energy[self.daily_flags.no_errors],
                        color='red')
            title += ', good days flagged'
        elif flag == 'bad':
            plt.scatter(xs[~self.daily_flags.no_errors],
                        energy[~self.daily_flags.no_errors],
                        color='red')
            title += ', bad days flagged'
        elif flag in ['clear', 'sunny']:
            plt.scatter(xs[self.daily_flags.clear],
                        energy[self.daily_flags.clear],
                        color='red')
            title += ', clear days flagged'
        elif flag == 'cloudy':
            plt.scatter(xs[self.daily_flags.clear],
                        energy[self.daily_flags.clear],
                        color='red')
            title += ', cloudy days flagged'
        plt.title(title)
        return fig

    def plot_clipping(self, figsize=(10, 8)):
        if self.daily_scores.clipping_1 is None:
            return
        fig, ax = plt.subplots(nrows=2, figsize=figsize, sharex=True)
        clip_stat_1 = self.daily_scores.clipping_1
        clip_stat_2 = self.daily_scores.clipping_2
        clipped_days = self.daily_flags.inverter_clipped
        ax[0].plot(clip_stat_1)
        ax[1].plot(clip_stat_2)
        if self.inverter_clipping:
            ax[0].scatter(np.arange(len(clip_stat_1))[clipped_days],
                          clip_stat_1[clipped_days], color='red', label='days with inverter clipping')
            ax[1].scatter(np.arange(len(clip_stat_2))[clipped_days],
                          clip_stat_2[clipped_days], color='red')
            ax[0].legend()
        ax[0].set_title('Clipping Score 1: ratio of daily max to overal max')
        ax[1].set_title('Clipping Score 2: fraction of time each day spent at daily max')
        return fig

    def plot_daily_max_pdf(self, figsize=(8, 6)):
        fig = self.__analyze_distribution(self.daily_scores.clipping_1[self.daily_flags.no_errors],
                                          plot='pdf', figsize=figsize)
        plt.title('Distribution of normalized daily maximum values')
        plt.legend()
        return fig

    def plot_daily_max_cdf(self, figsize=(10, 6)):
        fig = self.__analyze_distribution(self.daily_scores.clipping_1[self.daily_flags.no_errors],
                                          plot='cdf', figsize=figsize)
        plt.title('Cumulative density function of normalized daily maximum values')
        plt.legend()
        ax = plt.gca()
        ax.set_aspect('equal')
        return fig

    def plot_cdf_analysis(self, figsize=(12, 6)):
        fig = self.__analyze_distribution(self.daily_scores.clipping_1[self.daily_flags.no_errors],
                                          plot='diffs', figsize=figsize)
        return fig

    def __analyze_distribution(self, data, plot=None, figsize=(8, 6)):
        # Calculate empirical CDF
        x = np.sort(np.copy(data))
        x = x[x > 0]
        x = np.concatenate([[0.], x, [1.]])
        y = np.linspace(0, 1, len(x))
        # Resample the CDF to get an even spacing of points along the x-axis
        f = interp1d(x, y)
        x_rs = np.linspace(0, 1, 5000)
        y_rs = f(x_rs)
        # Fit statistical model to resampled CDF that has sparse 2nd order difference
        y_hat = cvx.Variable(len(y_rs))
        mu = cvx.Parameter(nonneg=True)
        mu.value = 1e1
        error = cvx.sum_squares(y_rs - y_hat)
        reg = cvx.norm(cvx.diff(y_hat, k=2), p=1)
        objective = cvx.Minimize(error + mu * reg)
        constraints = [
            y_rs[0] == y_hat[0],
            y[-1] == y_hat[-1]
        ]
        problem = cvx.Problem(objective, constraints)
        problem.solve(solver='MOSEK')
        # Look for outliers in the 2nd order difference to identify point masses from clipping
        local_curv = cvx.diff(y_hat, k=2).value
        ref_slope = cvx.diff(y_hat, k=1).value[:-1]
        threshold = -0.5
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
        point_masses = np.concatenate(
            [[False], metric <= threshold, # looking for drops of more than 65%
             [False]])
        # Catch if the PDF ends in a point mass at the high value
        if cvx.diff(y_hat, k=1).value[-1] > 5e-4:
            point_masses[-2] = True
        # Reduce clusters of detected points to single points
        pm_reduce = np.zeros_like(point_masses, dtype=np.bool)
        for ix in range(len(point_masses) - 1):
            if ~point_masses[ix] and point_masses[ix + 1]:
                begin_cluster = ix + 1
            elif point_masses[ix] and ~point_masses[ix + 1]:
                end_cluster = ix
                try:
                    ix_select = np.argmax(metric[begin_cluster:end_cluster + 1])
                except ValueError:
                    pm_reduce[begin_cluster] = True
                else:
                    pm_reduce[begin_cluster + ix_select] = True
        point_masses = pm_reduce
        point_mass_values = x_rs[point_masses]

        if plot is None:
            return point_mass_values
        elif plot == 'pdf':
            fig = plt.figure(figsize=figsize)
            plt.hist(data[data > 0], bins=100, alpha=0.5, label='histogram')
            scale = np.histogram(
                self.daily_scores.clipping_1[self.daily_scores.clipping_1 > 0],
                bins=100)[0].max() / cvx.diff(y_hat, k=1).value.max()
            plt.plot(x_rs[:-1], scale * cvx.diff(y_hat, k=1).value,
                     color='orange', linewidth=1, label='piecewise constant PDF estimate')
            for count, val in enumerate(point_mass_values):
                if count == 0:
                    plt.axvline(val, linewidth=1, linestyle=':',
                                color='green', label='detected point mass')
                else:
                    plt.axvline(val, linewidth=1, linestyle=':',
                                color='green')
            return fig
        elif plot == 'cdf':
            fig = plt.figure(figsize=figsize)
            plt.plot(x_rs, y_rs, linewidth=1, label='empirical CDF')
            plt.plot(x_rs, y_hat.value, linewidth=3, color='orange', alpha=0.57,
                     label='estimated CDF')
            if len(point_mass_values) > 0:
                plt.scatter(x_rs[point_masses], y_rs[point_masses],
                            color='red', marker='o',
                            label='detected point mass')
            return fig
        elif plot == 'diffs':
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



class DailyScores():
    def __init__(self):
        self.density = None
        self.linearity = None
        self.clipping_1 = None
        self.clipping_2 = None
        self.quality_clustering = None


class DailyFlags():
    def __init__(self):
        self.density = None
        self.linearity = None
        self.no_errors = None
        self.clear = None
        self.cloudy = None
        self.inverter_clipped = None

    def flag_no_errors(self):
        self.no_errors = np.logical_and(self.density, self.linearity)

    def flag_clear_cloudy(self, clear_days):
        self.clear = np.logical_and(clear_days, self.no_errors)
        self.cloudy = np.logical_and(~self.clear, self.no_errors)

class DailySignals():
    def __init__(self):
        self.density = None
        self.seasonal_density_fit = None
        self.energy = None