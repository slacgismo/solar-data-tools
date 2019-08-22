# -*- coding: utf-8 -*-
''' Data Handler Module

This module contains a class for managing a data processing pipeline

'''

import numpy as np
from scipy.stats import mode
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
        self.filled_data_matrix = None
        self.keys = None
        self.use_column = None
        self.capacity_estimate = None
        # Scores for the entire data set
        self.data_quality_score = None      # Fraction of days without data acquisition errors
        self.data_clearness_score = None    # Fraction of days that are approximately clear/sunny
        # Daily scores (floats) and flags (booleans)
        self.daily_scores = DailyScores()
        self.daily_flags = DailyFlags()
        # Useful daily signals defined by the data set
        self.daily_density_signal = None
        self.seasonal_density_fit = None
        self.daily_energy_signal = None
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
        self.score_data_set()
        return

    def make_data_matrix(self, use_col=None):
        df = standardize_time_axis(self.data_frame)
        if use_col is None:
            use_col = df.columns[0]
        self.raw_data_matrix = make_2d(df, key=use_col)
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
        self.daily_energy_signal = np.sum(self.filled_data_matrix, axis=0) *\
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
        self.daily_scores.density, self.daily_density_signal, self.seasonal_density_fit\
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
        if self.seasonal_density_fit is None:
            print('Run the density check first')
            return
        temp_mat = np.copy(self.filled_data_matrix)
        temp_mat[temp_mat < 0.005 * self.capacity_estimate] = np.nan
        difference_mat = np.round(temp_mat[1:] - temp_mat[:-1], 4)
        modes, counts = mode(difference_mat, axis=0, nan_policy='omit')
        n = self.filled_data_matrix.shape[0] - 1
        self.daily_scores.linearity = counts.data.squeeze() / (n * self.seasonal_density_fit)
        return

    def score_data_set(self):
        num_days = self.raw_data_matrix.shape[1]
        self.data_quality_score = np.sum(self.daily_flags.no_errors) / num_days
        self.data_clearness_score = np.sum(self.daily_flags.clear) / num_days
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
        if self.daily_density_signal is None:
            return
        fig = plt.figure(figsize=figsize)
        plt.plot(self.daily_density_signal, linewidth=1)
        xs = np.arange(len(self.daily_density_signal))
        title = 'Daily signal density'
        if flag == 'good':
            plt.scatter(xs[self.daily_flags.no_errors],
                        self.daily_density_signal[self.daily_flags.no_errors],
                        color='red')
            title += ', good days flagged'
        elif flag == 'bad':
            plt.scatter(xs[~self.daily_flags.no_errors],
                        self.daily_density_signal[~self.daily_flags.no_errors],
                        color='red')
            title += ', bad days flagged'
        elif flag in ['clear', 'sunny']:
            plt.scatter(xs[self.daily_flags.clear],
                        self.daily_density_signal[self.daily_flags.clear],
                        color='red')
            title += ', clear days flagged'
        elif flag == 'cloudy':
            plt.scatter(xs[self.daily_flags.cloudy],
                        self.daily_density_signal[self.daily_flags.cloudy],
                        color='red')
            title += ', cloudy days flagged'
        if np.logical_and(show_fit, self.seasonal_density_fit is not None):
            plt.plot(self.seasonal_density_fit, color='orange')
            plt.plot(0.6 * self.seasonal_density_fit, color='green', linewidth=1,
                     ls='--')
            plt.plot(1.05 * self.seasonal_density_fit, color='green', linewidth=1,
                     ls='--')
        plt.title(title)
        return fig


    def plot_daily_energy(self, flag=None, figsize=(8, 6)):
        if self.filled_data_matrix is None:
            return
        fig = plt.figure(figsize=figsize)
        energy = self.daily_energy_signal
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


class DailyScores():
    def __init__(self):
        self.density = None
        self.linearity = None


class DailyFlags():
    def __init__(self):
        self.density = None
        self.linearity = None
        self.no_errors = None
        self.clear = None
        self.cloudy = None

    def flag_no_errors(self):
        self.no_errors = np.logical_and(self.density, self.linearity)

    def flag_clear_cloudy(self, clear_days):
        self.clear = np.logical_and(clear_days, self.no_errors)
        self.cloudy = np.logical_and(~self.clear, self.no_errors)