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
import matplotlib.cm as cm
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from solardatatools.time_axis_manipulation import make_time_series,\
    standardize_time_axis, fix_time_shifts
from solardatatools.matrix_embedding import make_2d
from solardatatools.data_quality import daily_missing_data_advanced,\
    daily_missing_data_simple, dataset_quality_score
from solardatatools.data_filling import zero_nighttime, interp_missing
from solardatatools.clear_day_detection import find_clear_days
from solardatatools.plotting import plot_2d
from solardatatools.utilities import total_variation_plus_seasonal_quantile_filter

class DataHandler():
    def __init__(self, data_frame=None, raw_data_matrix=None,
                 convert_to_ts=False):
        self.data_frame = data_frame
        self.raw_data_matrix = raw_data_matrix
        if self.raw_data_matrix is not None:
            self.num_days = self.raw_data_matrix.shape[1]
            if self.raw_data_matrix.shape[0] <= 1400:
                self.data_sampling = int(24 * 60 / self.raw_data_matrix.shape[0])
            else:
                self.data_sampling = 24 * 60 / self.raw_data_matrix.shape[0]
        else:
            self.num_days = None
            self.data_sampling = None
        self.filled_data_matrix = None
        self.keys = None
        self.use_column = None
        self.capacity_estimate = None
        self.start_doy = None
        self.day_index = None
        # Scores for the entire data set
        self.data_quality_score = None      # Fraction of days without data acquisition errors
        self.data_clearness_score = None    # Fraction of days that are approximately clear/sunny
        # Flags for the entire data set
        self.inverter_clipping = None       # True if there is inverter clipping, false otherwise
        self.num_clip_points = None         # If clipping, the number of clipping set points
        self.capacity_changes = None        # True if the apparent capacity seems to change over the data set
        self.normal_quality_scores = None   # True if clustering of data quality scores are within decision boundaries
        self.time_shifts = None             # True if time shifts detected and corrected in data set
        # Daily scores (floats) and flags (booleans)
        self.daily_scores = DailyScores()
        self.daily_flags = DailyFlags()
        # Useful daily signals defined by the data set
        self.daily_signals = DailySignals()
        if np.alltrue([data_frame is not None, convert_to_ts]):
            df_ts, keys = make_time_series(self.data_frame)
            self.data_frame = df_ts
            self.keys = keys
        # Statistical clear sky fitting object
        self.scsf = None
        # Private attributes
        self._ran_pipeline = False
        self.__time_axis_standardized = False
        self.__density_lower_threshold = None
        self.__density_upper_threshold = None
        self.__linearity_threshold = None

    def run_pipeline(self, use_col=None, zero_night=True, interp_day=True,
                     fix_shifts=True, density_lower_threshold=0.6,
                     density_upper_threshold=1.05, linearity_threshold=0.1,
                     clear_tune_param=0.1, verbose=True, start_day_ix=None,
                     end_day_ix=None, c1=2., c2=500., estimator='com',
                     differentiate=False):
        t0 = time()
        if self.data_frame is not None:
            self.make_data_matrix(use_col, start_day_ix=start_day_ix,
                                  end_day_ix=end_day_ix,
                                     differentiate=differentiate)
        t1 = time()
        self.make_filled_data_matrix(zero_night=zero_night, interp_day=interp_day)
        t2 = time()
        self.capacity_estimate = np.quantile(self.filled_data_matrix, 0.95)
        if fix_shifts:
            self.auto_fix_time_shifts(c1=c1, c2=c2, estimator=estimator)
        t3 = time()
        try:
            self.get_daily_scores(threshold=0.2)
        except AttributeError:
            self.daily_scores = None
        t4 = time()
        try:
            self.get_daily_flags(density_lower_threshold=density_lower_threshold,
                                 density_upper_threshold=density_upper_threshold,
                                 linearity_threshold=linearity_threshold)
        except AttributeError:
            self.daily_scores = None
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
            out += 'fix time shifts: {:.2f}, \n'
            out += 'daily scores: {:.2f}, '
            out += 'daily flags: {:.2f}, '
            out += 'clear detect: {:.2f}, \n'
            out += 'clipping check: {:.2f}, '
            out += 'data scoring: {:.2f}'
            print(out.format(t8-t0, t1-t0, t2-t1, t3-t2, t4-t3, t5-t4, t6-t5, t7-t6, t8-t7))
        self._ran_pipeline = True
        return

    def report(self):
        try:
            l1 = 'Length:                {} days\n'.format(self.num_days)
            if self.raw_data_matrix.shape[0] <= 1440:
                l2 = 'Data sampling:         {} minute\n'.format(self.data_sampling)
            else:
                l2 = 'Data sampling:         {} second\n'.format(int(self.data_sampling * 60))
            l3 = 'Data quality score:    {:.1f}%\n'.format(self.data_quality_score * 100)
            l4 = 'Data clearness score:  {:.1f}%\n'.format(self.data_clearness_score * 100)
            l5 = 'Inverter clipping:     {}\n'.format(self.inverter_clipping)
            l6 = 'Time shifts corrected: {}'.format(self.time_shifts)
            p_out = l1 + l2 + l3 + l4 + l5 + l6
            print(p_out)
            if self.capacity_changes:
                print('WARNING: Changes in system capacity detected!')
            if self.num_clip_points > 1:
                print('WARNING: {} clipping set points detected!'.format(
                    self.num_clip_points
                ))
            if not self.normal_quality_scores:
                print('WARNING: Abnormal clustering of data quality scores!')
            return
        except TypeError:
            if self._ran_pipeline:
                m1 = 'Pipeline failed, please check data set.\n'
                m2 = "Suggest run .plot_heatmap(matrix='raw')\n\n"
                l1 = 'Length:                {} days\n'.format(self.num_days)
                if self.raw_data_matrix.shape[0] <= 1440:
                    l2 = 'Data sampling:         {} minute\n'.format(
                        self.data_sampling)
                else:
                    l2 = 'Data sampling:         {} second\n'.format(
                        int(self.data_sampling * 60))
                p_out = m1 + m2 + l1 + l2
                print(p_out)
            else:
                print('Please run the pipeline first!')
            return

    def make_data_matrix(self, use_col=None, start_day_ix=None, end_day_ix=None,
                                differentiate=False):

        if not self.__time_axis_standardized:
            df = standardize_time_axis(self.data_frame)
            self.data_frame = df
            self.__time_axis_standardized = True
        else:
            df = self.data_frame
        if use_col is None:
            use_col = df.columns[0]
        if differentiate:
            pass
        self.raw_data_matrix, day_index = make_2d(df, key=use_col, return_day_axis=True)
        self.raw_data_matrix = self.raw_data_matrix[:, start_day_ix:end_day_ix]
        self.num_days = self.raw_data_matrix.shape[1]
        if self.raw_data_matrix.shape[0] <= 1400:
            self.data_sampling = int(24 * 60 / self.raw_data_matrix.shape[0])
        else:
            self.data_sampling = 24 * 60 / self.raw_data_matrix.shape[0]
        self.use_column = use_col
        self.day_index = day_index[start_day_ix:end_day_ix]
        self.start_doy = self.day_index.dayofyear[0]
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
        day_counts = [np.logical_or(
            self.daily_scores.linearity[db.labels_ == lb] > linearity_threshold,
            np.logical_or(
                self.daily_scores.density[db.labels_ == lb] < density_lower_threshold,
                self.daily_scores.density[db.labels_ == lb] > density_upper_threshold
            )
        ) for lb in set(db.labels_)]
        self.normal_quality_scores = np.any([
            np.sum(day_count) <= max(5e-3 * self.num_days, 1)
            for day_count in day_counts
        ])
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
        temp_mat[temp_mat < 0.02 * self.capacity_estimate] = np.nan
        difference_mat = np.round(temp_mat[1:] - temp_mat[:-1], 4)
        modes, counts = mode(difference_mat, axis=0, nan_policy='omit')
        n = self.filled_data_matrix.shape[0] - 1
        self.daily_scores.linearity = counts.data.squeeze() / (n * self.daily_signals.seasonal_density_fit)
        return

    def score_data_set(self):
        num_days = self.raw_data_matrix.shape[1]
        try:
            self.data_quality_score = np.sum(self.daily_flags.no_errors) / num_days
        except TypeError:
            self.data_quality_score = None
        try:
            self.data_clearness_score = np.sum(self.daily_flags.clear) / num_days
        except TypeError:
            self.data_clearness_score = None
        try:
            self.capacity_clustering()
        except TypeError:
            self.capacity_changes = None
        return

    def clipping_check(self):
        max_value = np.max(self.filled_data_matrix)
        daily_max_val = np.max(self.filled_data_matrix, axis=0)
        # 1st clipping statistic: ratio of the max value on each day to overall max value
        clip_stat_1 = daily_max_val / max_value
        # 2nd clipping statistic: fraction of energy generated each day at or
        # near that day's max value
        with np.errstate(divide='ignore', invalid='ignore'):
            temp = self.filled_data_matrix / daily_max_val
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
            self.daily_flags.no_errors,
            clipped_days
        )
        # clipped days must also be near a peak in the distribution of the
        # 1st clipping statistic that shows the characteristic, strongly skewed
        # peak shape
        point_masses = self.__analyze_distribution(clip_stat_1)
        try:
            if len(point_masses) == 0:
                clipped_days[:] = False
            else:
                clipped_days[clipped_days] = np.any(
                    np.array([np.abs(clip_stat_1[clipped_days] - x0) < .02 for x0 in
                              point_masses]), axis=0
                )
        except IndexError:
            self.inverter_clipping = False
            self.num_clip_points = 0
            return
        self.daily_scores.clipping_1 = clip_stat_1
        self.daily_scores.clipping_2 = clip_stat_2
        self.daily_flags.inverter_clipped = clipped_days
        if np.sum(clipped_days) > 0.01 * self.num_days:
            self.inverter_clipping = True
            self.num_clip_points = len(point_masses)
        else:
            self.inverter_clipping = False
            self.num_clip_points = 0
        return

    def find_clipped_times(self):
        if self.inverter_clipping:
            daily_max_val = np.max(self.filled_data_matrix, axis=0)
            clip_stat_1 = self.daily_scores.clipping_1      #daily_max_val / max_value
            point_masses = self.__analyze_distribution(clip_stat_1)
            mat_normed = np.zeros_like(self.filled_data_matrix)
            msk = daily_max_val != 0
            mat_normed[:, msk] = self.filled_data_matrix[:, msk] / daily_max_val[msk]
            masks = np.stack([np.abs(mat_normed - x0) < 0.02
                              for x0 in point_masses])
            clipped_time_mask = np.alltrue(masks, axis=0)
            clipped_days = self.daily_flags.inverter_clipped
            clipped_time_mask[:, ~clipped_days] = False
            return clipped_time_mask
        else:
            return

    def capacity_clustering(self, plot=False, figsize=(8, 6),
                            show_clusters=True):
        if np.sum(self.daily_flags.no_errors) > 0:
            # Iterative reweighted L1 heuristic
            w = np.ones(len(self.daily_scores.clipping_1) - 1)
            eps = 0.5
            for i in range(5):
                s1, s2 = total_variation_plus_seasonal_quantile_filter(
                    self.daily_scores.clipping_1, self.daily_flags.no_errors,
                    tau=0.5, c1=15, c2=100,
                    c3=300, tv_weights=w
                )
                w = 1 / (eps + np.abs(np.diff(s1, n=1)))
        else:
            return
        db = DBSCAN(eps=.02, min_samples=max(0.1 * len(s1), 3)).fit(
            s1[:, np.newaxis]
        )
        if len(set(db.labels_)) > 1: #np.max(db.labels_) > 0:
            self.capacity_changes = True
        else:
            self.capacity_changes = False
        if plot:
            if show_clusters:
                fig, ax = plt.subplots(nrows=2, figsize=figsize, sharex=True,
                                       gridspec_kw={'height_ratios': [4, 1]})
                ax[0].plot(s1, label='capacity change detector')
                ax[0].plot(s2 + s1, label='signal model')
                ax[0].plot(self.daily_scores.clipping_1, alpha=0.3,
                         label='measured signal')
                ax[0].legend()
                ax[0].set_title('Detection of system capacity changes')
                ax[1].set_xlabel('day number')
                ax[0].set_ylabel('normalized daily maximum power')
                ax[1].plot(db.labels_, ls='none', marker='.')
                ax[1].set_ylabel('Capacity cluster label')
            else:
                fig, ax = plt.subplots(nrows=1, figsize=figsize)
                ax.plot(s1, label='capacity change detector')
                ax.plot(s2 + s1, label='signal model')
                ax.plot(self.daily_scores.clipping_1, alpha=0.3,
                         label='measured signal')
                ax.legend()
                ax.set_title('Detection of system capacity changes')
                ax.set_ylabel('normalized daily maximum power')
            return fig


    def auto_fix_time_shifts(self, c1=5., c2=500., estimator='com'):
        self.filled_data_matrix, shift_ixs = fix_time_shifts(
            self.filled_data_matrix, solar_noon_estimator=estimator, c1=c1, c2=c2,
            return_ixs=True, verbose=False, use_ixs=self.daily_flags.no_errors
        )
        if len(shift_ixs) == 0:
            self.time_shifts = False
        else:
            self.time_shifts = True

    def detect_clear_days(self, th=0.1):
        if self.filled_data_matrix is None:
            print('Generate a filled data matrix first.')
            return
        clear_days = find_clear_days(self.filled_data_matrix, th=th)
        self.daily_flags.flag_clear_cloudy(clear_days)
        return

    def find_clear_times(self):
        if self.scsf is None:
            print('No SCSF model detected. Fitting now...')
            self.fit_statistical_clear_sky_model()

    def fit_statistical_clear_sky_model(self, rank=6, mu_l=None, mu_r=None,
                                        tau=None, exit_criterion_epsilon=1e-3,
                                        max_iteration=10,
                                        calculate_degradation=True,
                                        max_degradation=None,
                                        min_degradation=None,
                                        non_neg_constraints=False,
                                        verbose=True, bootstraps=None):
        try:
            from statistical_clear_sky import SCSF
        except ImportError:
            print('Please install statistical-clear-sky package')
            return
        scsf = SCSF(data_handler_obj=self, rank_k=rank)
        scsf.execute(mu_l=mu_l, mu_r=mu_r, tau=tau,
                     exit_criterion_epsilon=exit_criterion_epsilon,
                     max_iteration=max_iteration,
                     is_degradation_calculated=calculate_degradation,
                     max_degradation=max_degradation,
                     min_degradation=min_degradation,
                     non_neg_constraints=non_neg_constraints,
                     verbose=verbose, bootstraps=bootstraps
                     )
        self.scsf = scsf

    def plot_heatmap(self, matrix='raw', flag=None, figsize=(12, 6),
                     scale_to_kw=False):
        if matrix == 'raw':
            mat = np.copy(self.raw_data_matrix)
        elif matrix == 'filled':
            mat = np.copy(self.filled_data_matrix)
        else:
            return
        if scale_to_kw:
            mat /= 1000
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
                           filled=True, ravel=True, figsize=(12, 6),
                           color=None, alpha=None):
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
        kwargs = {}
        if color is not None:
            kwargs['color'] = color
        if alpha is not None:
            kwargs['alpha'] = alpha
        plt.plot(plot_data, linewidth=1, **kwargs)
        return fig

    def plot_density_signal(self, flag=None, show_fit=False, figsize=(8, 6)):
        if self.daily_signals.density is None:
            return
        fig = plt.figure(figsize=figsize)
        try:
            xs = self.day_index.to_pydatetime()
        except AttributeError:
            xs = np.arange(len(self.daily_signals.density))
        plt.plot(xs, self.daily_signals.density, linewidth=1)
        title = 'Daily signal density'
        if flag == 'density':
            plt.plot(xs[~self.daily_flags.density],
                        self.daily_signals.density[~self.daily_flags.density],
                        ls='none', marker='.', color='red')
            title += ', density outlier days flagged'
        if flag == 'good':
            plt.plot(xs[self.daily_flags.no_errors],
                        self.daily_signals.density[self.daily_flags.no_errors],
                        ls='none', marker='.', color='red')
            title += ', good days flagged'
        elif flag == 'bad':
            plt.plot(xs[~self.daily_flags.no_errors],
                        self.daily_signals.density[~self.daily_flags.no_errors],
                        ls='none', marker='.', color='red')
            title += ', bad days flagged'
        elif flag in ['clear', 'sunny']:
            plt.plot(xs[self.daily_flags.clear],
                        self.daily_signals.density[self.daily_flags.clear],
                        ls='none', marker='.', color='red')
            title += ', clear days flagged'
        elif flag == 'cloudy':
            plt.plot(xs[self.daily_flags.cloudy],
                        self.daily_signals.density[self.daily_flags.cloudy],
                        ls='none', marker='.', color='red')
            title += ', cloudy days flagged'
        if np.logical_and(show_fit,
                          self.daily_signals.seasonal_density_fit is not None):
            plt.plot(xs, self.daily_signals.seasonal_density_fit, color='orange')
            plt.plot(xs, 0.6 * self.daily_signals.seasonal_density_fit,
                     color='green', linewidth=1,
                     ls='--')
            plt.plot(xs, 1.05 * self.daily_signals.seasonal_density_fit,
                     color='green', linewidth=1,
                     ls='--')
        plt.title(title)
        plt.gcf().autofmt_xdate()
        plt.ylabel('Fraction non-zero values')
        plt.xlabel('Date')
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

    def plot_daily_energy(self, flag=None, figsize=(8, 6), units='Wh'):
        if self.filled_data_matrix is None:
            return
        fig = plt.figure(figsize=figsize)
        energy = np.copy(self.daily_signals.energy)
        if np.max(energy) > 1000:
            energy /= 1000
            units = 'kWh'
        try:
            xs = self.day_index.to_pydatetime()
        except AttributeError:
            xs = np.arange(len(self.daily_signals.density))
        plt.plot(xs, energy, linewidth=1)
        title = 'Daily energy production'
        if flag == 'good':
            plt.plot(xs[self.daily_flags.no_errors],
                        energy[self.daily_flags.no_errors],
                        ls='none', marker='.', color='red')
            title += ', good days flagged'
        elif flag == 'bad':
            plt.plot(xs[~self.daily_flags.no_errors],
                        energy[~self.daily_flags.no_errors],
                        ls='none', marker='.', color='red')
            title += ', bad days flagged'
        elif flag in ['clear', 'sunny']:
            plt.plot(xs[self.daily_flags.clear],
                        energy[self.daily_flags.clear],
                        ls='none', marker='.', color='red')
            title += ', clear days flagged'
        elif flag == 'cloudy':
            plt.plot(xs[self.daily_flags.cloudy],
                        energy[self.daily_flags.cloudy],
                        ls='none', marker='.', color='red')
            title += ', cloudy days flagged'
        plt.title(title)
        plt.gcf().autofmt_xdate()
        plt.xlabel('Date')
        plt.ylabel('Energy ({})'.format(units))
        return fig

    def plot_clipping(self, figsize=(10, 8)):
        if self.daily_scores.clipping_1 is None:
            return
        fig, ax = plt.subplots(nrows=2, figsize=figsize, sharex=True)
        clip_stat_1 = self.daily_scores.clipping_1
        clip_stat_2 = self.daily_scores.clipping_2
        clipped_days = self.daily_flags.inverter_clipped
        try:
            xs = self.day_index.to_pydatetime()
        except AttributeError:
            xs = np.arange(len(self.daily_signals.density))
        ax[0].plot(xs, clip_stat_1)
        ax[1].plot(xs, clip_stat_2)
        if self.inverter_clipping:
            ax[0].plot(xs[clipped_days],
                          clip_stat_1[clipped_days], ls='none', marker='.',
                       color='red', label='days with inverter clipping')
            ax[1].plot(xs[clipped_days],
                          clip_stat_2[clipped_days], ls='none', marker='.',
                       color='red')
            ax[0].legend()
        ax[0].set_title('Clipping Score 1: ratio of daily max to overal max')
        ax[1].set_title('Clipping Score 2: fraction of daily energy generated at daily max power')
        ax[1].set_xlabel('Date')
        plt.gcf().autofmt_xdate()
        return fig

    def plot_daily_max_pdf(self, figsize=(8, 6)):
        fig = self.__analyze_distribution(self.daily_scores.clipping_1,
                                          plot='pdf', figsize=figsize)
        plt.title('Distribution of normalized daily maximum values')
        plt.xlabel('Normalized daily max power')
        plt.ylabel('Count')
        plt.legend()
        return fig

    def plot_daily_max_cdf(self, figsize=(10, 6)):
        fig = self.__analyze_distribution(self.daily_scores.clipping_1,
                                          plot='cdf', figsize=figsize)
        plt.title('Cumulative density function of\nnormalized daily maximum values')
        plt.xlabel('Normalized daily max power')
        plt.ylabel('Cumulative occurance probability')
        plt.legend()
        ax = plt.gca()
        ax.set_aspect('equal')
        return fig

    def plot_daily_max_cdf_and_pdf(self, figsize=(10, 6)):
        fig = self.__analyze_distribution(self.daily_scores.clipping_1,
                                          plot='both', figsize=figsize)
        return fig

    def plot_cdf_analysis(self, figsize=(12, 6)):
        fig = self.__analyze_distribution(self.daily_scores.clipping_1,
                                          plot='diffs', figsize=figsize)
        return fig

    def plot_capacity_change_analysis(self, figsize=(8, 6), show_clusters=True):
        fig = self.capacity_clustering(plot=True, figsize=figsize,
                                       show_clusters=show_clusters)
        return fig

    def plot_circ_dist(self, flag='good', num_bins=12*4, figsize=(8,8)):
        title = 'Calendar distribution of '
        if flag == 'good':
            slct = self.daily_flags.no_errors
            title += 'good days'
        elif flag == 'bad':
            slct = ~self.daily_flags.no_errors
            title += 'bad days'
        elif flag in ['clear', 'sunny']:
            slct = self.daily_flags.clear
            title += 'clear days'
        elif flag == 'cloudy':
            slct = self.daily_flags.cloudy
            title += 'cloudy days'
        circ_data = (self.start_doy + np.arange(self.num_days)[slct]) % 365 \
                    * 2 * np.pi / 365
        circ_hist = np.histogram(circ_data, bins=num_bins)
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
        start = (circ_hist[1][0] + circ_hist[1][1]) / 2
        end = (circ_hist[1][-1] + circ_hist[1][-2]) / 2
        theta = np.linspace(start, end, num_bins)
        radii = circ_hist[0]
        width = 2 * np.pi / num_bins
        bars = ax.bar(theta, radii, width=width, bottom=0.0, edgecolor='none')
        for r, bar in zip(radii, bars):
            bar.set_facecolor(cm.magma(r / np.max(circ_hist[0])))
            bar.set_alpha(0.75)
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_rorigin(-2.)
        ax.set_rlabel_position(0)
        ax.set_xticks(np.linspace(0, 2 * np.pi, 12, endpoint=False))
        ax.set_xticklabels(
            ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep',
             'Oct', 'Nov', 'Dec']
        )
        ax.set_title(title)
        # print(np.sum(circ_hist[0] <= 1))
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
            [[False], np.logical_and(metric <= threshold, ref_slope > 3e-4), # looking for drops of more than 65%
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
            scale = np.histogram(data[data > 0], bins=100)[0].max() \
                    / cvx.diff(y_hat, k=1).value.max()
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
        elif plot == 'both':
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
            ax1.set_xlabel('Cumulative occurance probability')
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
            ax2.legend(loc=(.15, .02))  #-0.4
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