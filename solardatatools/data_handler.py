# -*- coding: utf-8 -*-
''' Data Handler Module

This module contains a class for managing a data processing pipeline

'''

import numpy as np
import matplotlib.pyplot as plt
from solardatatools.time_axis_manipulation import make_time_series,\
    standardize_time_axis, fix_time_shifts
from solardatatools.matrix_embedding import make_2d
from solardatatools.data_quality import daily_missing_data_advanced,\
    daily_missing_data_simple, dataset_quality_score
from solardatatools.data_filling import zero_nighttime, interp_missing
from solardatatools.clear_day_detection import find_clear_days

class DataHandler():
    def __init__(self, data_frame=None, raw_data_matrix=None,
                 convert_to_ts=False):
        self.data_frame = data_frame
        self.raw_data_matrix = raw_data_matrix
        self.filled_data_matrix = None
        self.keys = None
        self.use_column = None
        self.data_score = None
        self.clear_day_score = None
        self.daily_density_flag = None
        self.daily_clear_flag = None
        self.density_signal = None
        self.daily_energy_signal = None
        if np.alltrue([data_frame is not None, convert_to_ts]):
            df_ts, keys = make_time_series(self.data_frame)
            self.data_frame = df_ts
            self.keys = keys

    def run_pipeline(self, use_col=None, zero_night=True, interp_day=True,
                     fix_shifts=True, threshold=0.2, use_advanced=True):
        if self.data_frame is not None:
            self.make_data_matrix(use_col)
        self.make_filled_data_matrix(zero_night=zero_night, interp_day=interp_day)
        if fix_shifts:
            self.auto_fix_time_shifts()
        self.run_density_check(threshold=threshold, use_advanced=use_advanced)
        self.detect_clear_days()
        return

    def make_data_matrix(self, use_col):
        df = standardize_time_axis(self.data_frame)
        self.raw_data_matrix = make_2d(df, key=use_col)
        self.use_column = use_col
        return

    def run_density_check(self, threshold=0.2, use_advanced=True):
        if self.raw_data_matrix is None:
            print('Generate a raw data matrix first.')
            return
        if use_advanced:
            self.daily_density_flag, self.density_signal = daily_missing_data_advanced(
                self.raw_data_matrix,
                threshold=threshold,
                return_density_signal=True
            )
        else:
            self.daily_density_flag, self.density_signal = daily_missing_data_simple(
                self.raw_data_matrix,
                threshold=threshold,
                return_density_signal=True
            )
        self.data_score = dataset_quality_score(self.raw_data_matrix,
                                                good_days=self.daily_density_flag)
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

    def auto_fix_time_shifts(self, c1=5., c2=500.):
        self.filled_data_matrix = fix_time_shifts(self.filled_data_matrix,
                                                  c1=c1, c2=c2)

    def detect_clear_days(self):
        if self.filled_data_matrix is None:
            print('Generate a filled data matrix first.')
            return
        self.daily_clear_flag = np.logical_and(
            find_clear_days(self.filled_data_matrix),
            self.daily_density_flag
        )

    def plot_density_signal(self, flag=None, figsize=(8, 6)):
        if self.density_signal is None:
            return
        fig = plt.figure(figsize=figsize)
        plt.plot(self.density_signal)
        xs = np.arange(len(self.density_signal))
        if flag == 'good':
            plt.scatter(xs[self.daily_density_flag],
                        self.density_signal[self.daily_density_flag],
                        color='red')
        elif flag == 'bad':
            plt.scatter(xs[~self.daily_density_flag],
                        self.density_signal[~self.daily_density_flag],
                        color='red')
        elif flag == 'sunny':
            plt.scatter(xs[self.daily_clear_flag],
                        self.density_signal[self.daily_clear_flag],
                        color='red')
        elif flag == 'cloudy':
            msk = np.logical_and(
                ~self.daily_clear_flag,
                self.daily_density_flag
            )
            plt.scatter(xs[msk],
                        self.density_signal[msk],
                        color='red')
        return fig


    def plot_daily_energy(self, flag=None, figsize=(8, 6)):
        if self.filled_data_matrix is None:
            return
        fig = plt.figure(figsize=figsize)
        energy = self.daily_energy_signal
        plt.plot(energy)
        xs = np.arange(len(energy))
        if flag == 'good':
            plt.scatter(xs[self.daily_density_flag],
                        energy[self.daily_density_flag],
                        color='red')
        elif flag == 'bad':
            plt.scatter(xs[~self.daily_density_flag],
                        energy[~self.daily_density_flag],
                        color='red')
        elif flag == 'sunny':
            plt.scatter(xs[self.daily_clear_flag],
                        energy[self.daily_clear_flag],
                        color='red')
        elif flag == 'cloudy':
            msk = np.logical_and(
                ~self.daily_clear_flag,
                self.daily_density_flag
            )
            plt.scatter(xs[msk],
                        energy[msk],
                        color='red')
        return fig
