# -*- coding: utf-8 -*-
''' Data Handler Module

This module contains a class for managing a data processing pipeline

'''

import numpy as np
from scipy.stats import mode
from scipy.signal import argrelextrema
from sklearn.neighbors.kde import KernelDensity
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
        self.capacity_changes = None        # True if the apparent capacity seems to change over the data set
        # Daily scores (floats) and flags (booleans)
        self.daily_scores = DailyScores()
        self.daily_flags = DailyFlags()
        # Useful daily signals defined by the data set
        self.daily_signals = DailySignals()
        if np.alltrue([data_frame is not None, convert_to_ts]):
            df_ts, keys = make_time_series(self.data_frame)
            self.data_frame = df_ts
            self.keys = keys

    def run_pipeline(self, use_col=None, zero_night=True, interp_day=True,
                     fix_shifts=True, density_lower_threshold=0.6,
                     density_upper_threshold=1.05, linearity_threshold=0.1):
        if self.data_frame is not None:
            self.make_data_matrix(use_col)
        self.make_filled_data_matrix(zero_night=zero_night, interp_day=interp_day)
        self.capacity_estimate = np.quantile(self.filled_data_matrix, 0.95)
        if fix_shifts:
            self.auto_fix_time_shifts()
        self.get_daily_scores(threshold=0.2)
        self.get_daily_flags(density_lower_threshold=density_lower_threshold,
                             density_upper_threshold=density_upper_threshold,
                             linearity_threshold=linearity_threshold)
        self.detect_clear_days()
        self.clipping_check()
        self.score_data_set()
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
            return
        except TypeError:
            print('Please run the pipeline first!')
            return

    def make_data_matrix(self, use_col=None):
        df = standardize_time_axis(self.data_frame)
        if use_col is None:
            use_col = df.columns[0]
        self.raw_data_matrix = make_2d(df, key=use_col)
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
        # 2nd clipping statistic: fraction of time each day spent at that day's max value
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
        # 1st clipping statistic
        peak_locs, peak_vals = self.__analyze_distribution(clip_stat_1)
        clipped_days[clipped_days] = np.any(
            np.array([np.abs(clip_stat_1[clipped_days] - x0) < .05 for x0 in
                      peak_locs]), axis=0
        )
        self.daily_scores.clipping_1 = clip_stat_1
        self.daily_scores.clipping_2 = clip_stat_2
        self.daily_flags.inverter_clipped = clipped_days
        if np.sum(clipped_days) > 0:
            self.inverter_clipping = True
        else:
            self.inverter_clipping = False
        return


    def auto_fix_time_shifts(self, c1=5., c2=500.):
        self.filled_data_matrix = fix_time_shifts(self.filled_data_matrix,
                                                  c1=c1, c2=c2)

    def detect_clear_days(self):
        if self.filled_data_matrix is None:
            print('Generate a filled data matrix first.')
            return
        clear_days = find_clear_days(self.filled_data_matrix)
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

    def plot_density_signal(self, flag=None, show_fit=False, figsize=(8, 6)):
        if self.daily_signals.density is None:
            return
        fig = plt.figure(figsize=figsize)
        plt.plot(self.daily_signals.density, linewidth=1)
        xs = np.arange(len(self.daily_signals.density))
        title = 'Daily signal density'
        if flag == 'good':
            plt.scatter(xs[self.daily_flags.no_errors],
                        self.daily_signals.density[self.daily_flags.no_errors],
                        color='red')
            title += ', good days flagged'
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
        if np.logical_and(show_fit, self.daily_signals.seasonal_density_fit is not None):
            plt.plot(self.daily_signals.seasonal_density_fit, color='orange')
            plt.plot(0.6 * self.daily_signals.seasonal_density_fit, color='green', linewidth=1,
                     ls='--')
            plt.plot(1.05 * self.daily_signals.seasonal_density_fit, color='green', linewidth=1,
                     ls='--')
        plt.title(title)
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
        ax[0].scatter(np.arange(len(clip_stat_1))[clipped_days],
                      clip_stat_1[clipped_days], color='red', label='days with inverter clipping')
        ax[1].scatter(np.arange(len(clip_stat_2))[clipped_days],
                      clip_stat_2[clipped_days], color='red')
        ax[0].set_title('Clipping Score 1: ratio of daily max to overal max')
        ax[1].set_title('Clipping Score 2: fraction of time each day spent at daily max')
        ax[0].legend()
        return fig

    def plot_daily_max_distribution(self, figsize=(8, 6)):
        fig = self.__analyze_distribution(self.daily_scores.clipping_1,
                                          plot=True, figsize=figsize)
        plt.title('Distribution of normalized daily maximum values')
        plt.legend()
        return fig

    def __analyze_distribution(self, data, plot=False, figsize=(8, 6)):
        # set the bandwidth for the KDE algorithm dynamically as a logarithmic
        # function of the number of values. The function roughly follows the
        # following:
        #
        # data length  |  bandwidth
        # -------------|------------
        #     10       |     0.08
        #     50       |     0.05
        #     500      |     0.025
        #     2000     |     0.01
        coeffs = np.array(
            [1.28782573e+01 / 1000, 2.99960708e-07, -8.76881301e+01 / 1000])

        def bdw(x):
            out = coeffs[0] * -np.log(coeffs[1] * x) + coeffs[2]
            return np.clip(out, 0.01, 0.1)

        bandwidth = bdw(len(data))

        kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(
            data[data > 0][:, np.newaxis])
        X_plot = np.linspace(np.min(data) - 0.01,
                             np.max(data) + 0.01)[:, np.newaxis]
        log_dens = kde.score_samples(X_plot)
        mins = argrelextrema(log_dens, np.less)[0]  # potential cut points to make clusters
        maxs = argrelextrema(log_dens, np.greater)[0]  # locations of the max point in each cluster
        if len(mins) >= len(maxs):
            if mins[0] < maxs[0]:
                mins = mins[1:]
            if mins[-1] > maxs[-1]:
                mins = mins[:-1]

        keep_mxs = np.ones_like(maxs, dtype=np.bool)
        keep_mns = np.ones_like(mins, dtype=np.bool)
        index = np.arange(len(maxs))
        done = False
        if len(mins) > 0:
            while not done:
                comp_left = np.exp(log_dens[maxs[keep_mxs][:-1]]) - np.exp(
                    log_dens[mins[keep_mns]])
                comp_right = np.exp(log_dens[maxs[keep_mxs][1:]]) - np.exp(
                    log_dens[mins[keep_mns]])
                comp_array = np.c_[comp_left, comp_right]
                min_diff = comp_array.min()
                if min_diff < 0.35:
                    min_dif_loc = np.unravel_index(comp_array.argmin(),
                                                   comp_array.shape)
                    drop_min = min_dif_loc[0]
                    keep_mns[mins == mins[keep_mns][drop_min]] = 0
                    if np.exp(log_dens[maxs[keep_mxs][drop_min]]) > np.exp(
                            log_dens[maxs[keep_mxs][drop_min + 1]]):
                        keep_mxs[maxs == maxs[keep_mxs][drop_min + 1]] = 0
                    else:
                        keep_mxs[maxs == maxs[keep_mxs][drop_min]] = 0
                else:
                    done = True
                if np.sum(keep_mns) == 0:
                    done = True

        if plot:
            fig = plt.figure(figsize=figsize)
            plt.hist(data, bins=100)
            plt.plot(X_plot.squeeze(),
                     0.01 * len(data) * np.exp(log_dens), label='KDE fit')
            for ix, mn in enumerate(mins[keep_mns]):
                if ix == 0:
                    plt.axvline(X_plot[:, 0][mn], linewidth=1, linestyle=':',
                                color='green', label='detected minimum')
                else:
                    plt.axvline(X_plot[:, 0][mn], linewidth=1, linestyle=':',
                                color='green')
            for ix, mx in enumerate(maxs[keep_mxs]):
                if ix == 0:
                    plt.axvline(X_plot[:, 0][mx], linewidth=1, linestyle='--',
                                color='red', label='detected maximum')
                else:
                    plt.axvline(X_plot[:, 0][mx], linewidth=1, linestyle='--',
                                color='red')
            return fig
        else:
            peak_locs = X_plot[:, 0][maxs[keep_mxs]]
            peak_vals = np.exp(kde.score_samples(peak_locs[:, np.newaxis]))
            return peak_locs, peak_vals


class DailyScores():
    def __init__(self):
        self.density = None
        self.linearity = None
        self.clipping_1 = None
        self.clipping_2 = None


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