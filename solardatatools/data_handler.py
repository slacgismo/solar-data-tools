# -*- coding: utf-8 -*-
''' Data Handler Module

This module contains a class for managing a data processing pipeline

'''
from time import time
from datetime import timedelta
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.stats import mode, skew
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import traceback, sys
from solardatatools.time_axis_manipulation import make_time_series,\
    standardize_time_axis
from solardatatools.matrix_embedding import make_2d
from solardatatools.data_quality import daily_missing_data_advanced
from solardatatools.data_filling import zero_nighttime, interp_missing
from solardatatools.clear_day_detection import find_clear_days
from solardatatools.plotting import plot_2d
from solardatatools.clear_time_labeling import find_clear_times
from solardatatools.solar_noon import avg_sunrise_sunset
from solardatatools.algorithms import CapacityChange, TimeShift,\
    SunriseSunset, ClippingDetection

class DataHandler():
    def __init__(self, data_frame=None, raw_data_matrix=None, datetime_col=None,
                 convert_to_ts=False, no_future_dates=True,
                 aggregate=None, how=lambda x: x.mean()):
        if data_frame is not None:
            if convert_to_ts:
                data_frame, keys = make_time_series(data_frame)
                self.keys = keys
            else:
                self.keys = list(data_frame.columns)
            self.data_frame_raw = data_frame.copy()
            seq_index = np.arange(len(self.data_frame_raw))
            if isinstance(self.keys[0], tuple) and not convert_to_ts:
                num_levels = len(self.keys[0])
                self.seq_index_key = tuple(['seq_index'] * num_levels)
            else:
                self.seq_index_key = 'seq_index'
            self.data_frame_raw[self.seq_index_key] = seq_index
            if not isinstance(self.data_frame_raw.index, pd.DatetimeIndex):
                if datetime_col is not None:
                    df = self.data_frame_raw
                    df[datetime_col] = pd.to_datetime(df[datetime_col])
                    df.set_index(datetime_col, inplace=True)
                else:
                    e = "Data frame must have a DatetimeIndex or"
                    e += "the user must set the datetime_col kwarg."
                    raise Exception(e)
            df_index = self.data_frame_raw.index
            if df_index.tz is not None:
                df_index = df_index.tz_localize(None)
            if no_future_dates:
                now = datetime.now()
                self.data_frame_raw = self.data_frame_raw[
                    self.data_frame_raw.index <= now
                ]
            self.data_frame = None
            if aggregate is not None:
                new_data = how(self.data_frame_raw.resample(aggregate))
                self.data_frame_raw = new_data
        else:
            self.data_frame_raw = None
            self.data_frame = None
            self.keys = None
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
        self.use_column = None
        self.capacity_estimate = None
        self.start_doy = None
        self.day_index = None
        self.power_units = None
        # "Extra" data, i.e. additional columns to process from the table
        self.extra_matrices = {}            # Matrix views of extra columns
        self.extra_quality_scores = {}      # Relative quality: fraction of non-NaN values in column during daylight time periods, as defined by the main power columns
        # Scores for the entire data set
        self.data_quality_score = None      # Fraction of days without data acquisition errors
        self.data_clearness_score = None    # Fraction of days that are approximately clear/sunny
        # Flags for the entire data set
        self.inverter_clipping = None       # True if there is inverter clipping, false otherwise
        self.num_clip_points = None         # If clipping, the number of clipping set points
        self.capacity_changes = None        # True if the apparent capacity seems to change over the data set
        self.normal_quality_scores = None   # True if clustering of data quality scores are within decision boundaries
        self.time_shifts = None             # True if time shifts detected and corrected in data set
        self.tz_correction = 0              # TZ correction factor (determined during pipeline run)
        # Daily scores (floats), flags (booleans), and boolean masks
        self.daily_scores = DailyScores()   # 1D arrays of floats
        self.daily_flags = DailyFlags()     # 1D arrays of Booleans
        self.boolean_masks = BooleanMasks() # 2D arrays of Booleans
        # Useful daily signals defined by the data set
        self.daily_signals = DailySignals()
        # Algorithm objects
        self.scsf = None
        self.capacity_analysis = None
        self.time_shift_analysis = None
        self.daytime_analysis = None
        self.clipping_analysis = None
        # Private attributes
        self._ran_pipeline = False
        self._error_msg = ''
        self.__density_lower_threshold = None
        self.__density_upper_threshold = None
        self.__linearity_threshold = None
        self.__recursion_depth = 0
        self.__initial_time = None
        self.__fix_dst_ran = False

    def run_pipeline(self, power_col=None, min_val=-5, max_val=None,
                     zero_night=True, interp_day=True, fix_shifts=True,
                     density_lower_threshold=0.6, density_upper_threshold=1.05,
                     linearity_threshold=0.1, clear_day_smoothness_param=0.9,
                     clear_day_energy_param=0.8, verbose=True,
                     start_day_ix=None, end_day_ix=None, c1=None, c2=500.,
                     solar_noon_estimator='com', correct_tz=True, extra_cols=None,
                     daytime_threshold=0.1, units='W', solver=None):
        self.daily_scores = DailyScores()
        self.daily_flags = DailyFlags()
        self.capacity_analysis = None
        self.time_shift_analysis = None
        self.extra_matrices = {}            # Matrix views of extra columns
        self.extra_quality_scores = {}
        self.power_units = units
        if self.__recursion_depth == 0:
            self.tz_correction = 0
        t = np.zeros(6)
        ######################################################################
        # Preprocessing
        ######################################################################
        t[0] = time()
        # If power_col not passed, assume that the first column contains the
        # data to be processed
        if power_col is None:
            power_col = self.data_frame_raw.columns[0]
        if power_col not in self.data_frame_raw.columns:
            print('Power column key not present in data frame.')
            return
        # Pandas operations to make a time axis with regular intervals.
        # If correct_tz is True, it will also align the median daily maximum
        if self.data_frame_raw is not None:
            self.data_frame, sn_deviation = standardize_time_axis(
                self.data_frame_raw, timeindex=True, power_col=power_col,
                correct_tz=correct_tz, verbose=verbose
            )
            if correct_tz:
                self.tz_correction = sn_deviation
        # Embed the data as a matrix, with days in columns. Also, set some
        # attributes, like the scan rate, day index, and day of year arary.
        # Almost never use start_day_ix and end_day_ix, but they're there
        # if a user wants to use a portion of the data set.
        if self.data_frame is not None:
            self.make_data_matrix(power_col, start_day_ix=start_day_ix,
                                  end_day_ix=end_day_ix)

        if max_val is not None:
            mat_copy = np.copy(self.raw_data_matrix)
            mat_copy[np.isnan(mat_copy)] = -9999
            slct = mat_copy > max_val
            if np.sum(slct) > 0:
                self.raw_data_matrix[slct] = np.nan
        if min_val is not None:
            mat_copy = np.copy(self.raw_data_matrix)
            mat_copy[np.isnan(mat_copy)] = 9999
            slct = mat_copy < min_val
            if np.sum(slct) > 0:
                self.raw_data_matrix[slct] = np.nan
        self.capacity_estimate = np.nanquantile(self.raw_data_matrix, 0.95)
        if self.capacity_estimate <= 500 and self.power_units == 'W':
            self.power_units = 'kW'
        self.boolean_masks.missing_values = np.isnan(self.raw_data_matrix)
        # Run once to get a rough estimate. Update at the end after cleaning
        # is finished
        ss = SunriseSunset()
        # CVXPY - either MOSEK or ECOS for this one, SCS fails
        try:
            if solver is None or solver == 'MOSEK':
                ss.run_optimizer(self.raw_data_matrix, plot=False, solver=solver)
            else:
                ss.run_optimizer(self.raw_data_matrix, plot=False, solver='ECOS')
            self.boolean_masks.daytime = ss.sunup_mask_estimated
        except:
            msg = 'Sunrise/sunset detection failed.'
            self._error_msg += '\n' + msg
            if verbose:
                print(msg)
                traceback.print_exception(*sys.exc_info())
        self.daytime_analysis = ss
        ######################################################################
        # Cleaning
        ######################################################################
        t[1] = time()
        try:
            self.make_filled_data_matrix(zero_night=zero_night,
                                         interp_day=interp_day)
        except:
            msg = 'Matrix filling failed.'
            self._error_msg += '\n' + msg
            if verbose:
                print(msg)
                traceback.print_exception(*sys.exc_info())
        num_raw_measurements = np.count_nonzero(
            np.nan_to_num(self.raw_data_matrix,
                          copy=True,
                          nan=0.)[self.boolean_masks.daytime]
        )
        num_filled_measurements = np.count_nonzero(
            np.nan_to_num(self.filled_data_matrix,
                          copy=True,
                          nan=0.)[self.boolean_masks.daytime]
        )
        if num_raw_measurements > 0:
            ratio = num_filled_measurements / num_raw_measurements
        else:
            msg = 'Error: data set contains no non-zero values!'
            self._error_msg += '\n' + msg
            if verbose:
                print(msg)
            self.daily_scores = None
            self.daily_flags = None
            self.data_quality_score = 0.0
            self.data_clearness_score = 0.0
            self._ran_pipeline = True
            return
        if ratio < 0.9:
            msg = 'Error: data was lost during NaN filling procedure. '
            msg += 'This typically occurs when\nthe time stamps are in the '
            msg += 'wrong timezone. Please double check your data table.\n'
            self._error_msg += '\n' + msg
            if verbose:
                print(msg)
            self.daily_scores = None
            self.daily_flags = None
            self.data_quality_score = None
            self.data_clearness_score = None
            self._ran_pipeline = True
            return
        ######################################################################
        # Scoring
        ######################################################################
        t[2] = time()
        t_clean = np.zeros(6)
        t_clean[0] = time()
        try:
            # CVXPY - density scoring
            self.get_daily_scores(threshold=0.2, solver=solver)
        except:
            msg = 'Daily quality scoring failed.'
            self._error_msg += '\n' + msg
            if verbose:
                print(msg)
                traceback.print_exception(*sys.exc_info())
            self.daily_scores = None
        try:
            self.get_daily_flags(density_lower_threshold=density_lower_threshold,
                                 density_upper_threshold=density_upper_threshold,
                                 linearity_threshold=linearity_threshold)
        except:
            msg = 'Daily quality flagging failed.'
            self._error_msg += '\n' + msg
            if verbose:
                print(msg)
                traceback.print_exception(*sys.exc_info())
            self.daily_flags = None
        t_clean[1] = time()
        try:
            # CVXPY
            self.detect_clear_days(
                smoothness_threshold=clear_day_smoothness_param,
                energy_threshold=clear_day_energy_param,
                solver=solver
            )
        except:
            msg = 'Clear day detection failed.'
            self._error_msg += '\n' + msg
            if verbose:
                print(msg)
                traceback.print_exception(*sys.exc_info())
        t_clean[2] = time()
        try:
            # CVXPY
            self.clipping_check(solver=solver)
        except Exception as e:
            msg = 'clipping check failed: ' + str(e)
            self._error_msg += '\n' + msg
            if verbose:
                print(msg)
                traceback.print_exception(*sys.exc_info())
            self.inverter_clipping = None
        t_clean[3] = time()
        try:
            self.score_data_set()
        except:
            msg = 'Data set summary scoring failed.'
            self._error_msg += '\n' + msg
            if verbose:
                print(msg)
                traceback.print_exception(*sys.exc_info())
            self.data_quality_score = None
            self.data_clearness_score = None
        t_clean[4] = time()
        try:
            # CVXPY
            self.capacity_clustering(solver=solver)
        except TypeError:
            msg = 'Capacity clustering failed.'
            self._error_msg += '\n' + msg
            if verbose:
                print(msg)
                traceback.print_exception(*sys.exc_info())
            self.capacity_changes = None
        t_clean[5] = time()
        ######################################################################
        # Remaining data cleaning operations
        # Fix time shifts depends on the data clearness scoring, and the other
        # two depend on fixing the time shifts, when then occur.
        ######################################################################
        t[3] = time()
        if fix_shifts:
            try:
                # CVXPY
                self.auto_fix_time_shifts(c1=c1, c2=c2,
                                          estimator=solar_noon_estimator,
                                          threshold=daytime_threshold,
                                          periodic_detector=False,
                                          solver=solver)
            except Exception as e:
                msg = 'Fix time shift algorithm failed.'
                self._error_msg += '\n' + msg
                if verbose:
                    print(msg)
                    print('Error message:', e)
                    print('\n')
                    traceback.print_exception(*sys.exc_info())
                self.time_shifts = None

        # check for remaining TZ offset issues
        if correct_tz:
            average_noon = np.nanmean(
                avg_sunrise_sunset(self.filled_data_matrix, threshold=0.01)
            )
            tz_offset = int(np.round(12 - average_noon))
            if np.abs(tz_offset) > 1:
                self.tz_correction += tz_offset
                # Related to this bug fix:
                # https://github.com/slacgismo/solar-data-tools/commit/ae0037771c09ace08bff5a4904475da606e934da
                old_index = self.data_frame.index.copy()
                self.data_frame.index = self.data_frame.index.shift(
                    tz_offset, freq='H'
                )
                self.data_frame = self.data_frame.reindex(index=old_index,
                                                          method='nearest',
                                                          limit=1).fillna(0)
                meas_per_hour = self.filled_data_matrix.shape[0] / 24
                roll_by = int(meas_per_hour * tz_offset)
                self.filled_data_matrix = np.nan_to_num(
                    np.roll(self.filled_data_matrix, roll_by, axis=0),
                    0
                )
                self.raw_data_matrix = np.roll(
                    self.raw_data_matrix, roll_by, axis=0
                )
                self.boolean_masks.daytime = np.roll(
                    self.boolean_masks.daytime, roll_by, axis=0
                )

        # Update daytime detection based on cleaned up data
        # self.daytime_analysis.run_optimizer(self.filled_data_matrix, plot=False)
        # CVXPY
        self.daytime_analysis.calculate_times(self.filled_data_matrix,
                                              solver=solver)
        self.boolean_masks.daytime = self.daytime_analysis.sunup_mask_estimated
        ######################################################################
        # Process Extra columns
        ######################################################################
        t[4] = time()
        if extra_cols is not None:
            freq = int(self.data_sampling * 60)
            new_index = pd.date_range(start=self.day_index[0].date(),
                                      end=self.day_index[-1].date() + timedelta(
                                          days=1),
                                      freq='{}s'.format(freq))[:-1]
            if isinstance(extra_cols, str):
                extra_cols = np.atleast_1d(extra_cols)
            elif isinstance(extra_cols, tuple):
                extra_cols = [extra_cols]
            for col in extra_cols:
                self.generate_extra_matrix(col, new_index=new_index)
        t[5] = time()
        times = np.diff(t, n=1)
        cleaning_times = np.diff(t_clean, n=1)
        total_time = t[-1] - t[0]
        # Cleanup
        self.__recursion_depth = 0
        if verbose:
            if self.__initial_time is not None:
                restart_msg = '{:.2f} seconds spent automatically localizing the time zone\n'
                restart_msg += 'Info for last pipeline run below:\n'
                restart_msg = restart_msg.format(t[0] - self.__initial_time)
                print(restart_msg)
            out = 'total time: {:.2f} seconds\n'
            out += '--------------------------------\n'
            out += 'Breakdown\n'
            out += '--------------------------------\n'
            out += 'Preprocessing              {:.2f}s\n'
            out += 'Cleaning                   {:.2f}s\n'
            out += 'Filtering/Summarizing      {:.2f}s\n'
            out += '    Data quality           {:.2f}s\n'
            out += '    Clear day detect       {:.2f}s\n'
            out += '    Clipping detect        {:.2f}s\n'
            out += '    Capacity change detect {:.2f}s\n'
            if extra_cols is not None:
                out += 'Extra Column Processing    {:.2f}s'
            print(out.format(
                total_time,
                times[0],
                times[1] + times[3],
                times[2],
                cleaning_times[0],
                cleaning_times[1],
                cleaning_times[2],
                cleaning_times[4],
                times[4]
            ))
        self._ran_pipeline = True
        return

    def report(self):
        try:
            if self.num_days >= 365:
                l1 = 'Length:                {:.2f} years\n'.format(self.num_days / 365)
            else:
                l1 = 'Length:                {} days\n'.format(self.num_days)
            if self.power_units == 'W':
                l1_a = 'Capacity estimate:     {:.2f} kW\n'.format(self.capacity_estimate / 1000)
            elif self.power_units == 'kW':
                l1_a = 'Capacity estimate:     {:.2f} kW\n'.format(self.capacity_estimate)
            else:
                l1_a = 'Capacity estimate:     {:.2f} '.format(self.capacity_estimate)
                l1_a += self.power_units + '\n'
            if self.raw_data_matrix.shape[0] <= 1440:
                l2 = 'Data sampling:         {} minute\n'.format(self.data_sampling)
            else:
                l2 = 'Data sampling:         {} second\n'.format(int(self.data_sampling * 60))
            l3 = 'Data quality score:    {:.1f}%\n'.format(self.data_quality_score * 100)
            l4 = 'Data clearness score:  {:.1f}%\n'.format(self.data_clearness_score * 100)
            l5 = 'Inverter clipping:     {}\n'.format(self.inverter_clipping)
            l6 = 'Time shifts corrected: {}\n'.format(self.time_shifts)
            if self.tz_correction != 0:
                l7 = 'Time zone correction:  {} hours'.format(int(self.tz_correction))
            else:
                l7 = 'Time zone correction:  None'
            p_out = l1 + l1_a + l2 + l3 + l4 + l5 + l6 + l7
            if self.capacity_changes:
                p_out += '\nWARNING: Changes in system capacity detected!'
            if self.num_clip_points > 1:
                p_out += '\nWARNING: {} clipping set points detected!'.format(
                    self.num_clip_points
                )
            if not self.normal_quality_scores:
                p_out += '\nWARNING: Abnormal clustering of data quality scores!'
            print(p_out)
            return
        except TypeError:
            if self._ran_pipeline:
                m1 = 'Pipeline failed, please check data set.\n'
                m2 = "Try running: self.plot_heatmap(matrix='raw')\n\n"
                if self.num_days >= 365:
                    l1 = 'Length:                {:.2f} years\n'.format(
                        self.num_days / 365)
                else:
                    l1 = 'Length:                {} days\n'.format(
                        self.num_days)
                if self.power_units == 'W':
                    l1_a = 'Capacity estimate:     {:.2f} kW\n'.format(self.capacity_estimate / 1000)
                elif self.power_units == 'kW':
                    l1_a = 'Capacity estimate:     {:.2f} kW\n'.format(self.capacity_estimate)
                else:
                    l1_a = 'Capacity estimate:     {:.2f} '.format(self.capacity_estimate)
                    l1_a += self.power_units + '\n'
                if self.raw_data_matrix.shape[0] <= 1440:
                    l2 = 'Data sampling:         {} minute\n'.format(
                        self.data_sampling)
                else:
                    l2 = 'Data sampling:         {} second\n'.format(
                        int(self.data_sampling * 60))
                p_out = m1 + m2 + l1 + l1_a + l2
                print(p_out)
                print('\nError messages captured from pipeline:' + self._error_msg)
            else:
                print('Please run the pipeline first!')
            return

    def augment_data_frame(self, boolean_index, column_name):
        """
        Add a column to the data frame (tabular) representation of the data,
        containing True/False values at each time stamp.
        Boolean index is a 1-D or 2-D numpy array of True/False values. If 1-D,
        array should be of length N, where N is the number of days in the data
        set. If 2-D, the array should be of size M X N where M is the number
        of measurements each day and N is the number of days.

        :param boolean_index: Length N or size M X N numpy arrays of booleans
        :param column_name: Name for column
        :return:
        """
        if self.data_frame is None:
            print('This DataHandler object does not contain a data frame.')
            return
        if boolean_index is None:
            print('No mask available for ' + column_name)
            return
        m, n = self.raw_data_matrix.shape
        index_shape = boolean_index.shape
        cond1 = index_shape == (m, n)
        cond2 = index_shape == (n ,)
        if not cond1 and not cond2:
            print('Boolean index shape does not match the data.')
        elif cond1:
            if self.time_shifts:
                ts = self.time_shift_analysis
                boolean_index = ts.invert_corrections(boolean_index)
            start = self.day_index[0]
            freq = '{}min'.format(self.data_sampling)
            periods = self.filled_data_matrix.size
            tindex = pd.date_range(start=start, freq=freq, periods=periods)
            series = pd.Series(data=boolean_index.ravel(order='F'), index=tindex)
            series.name = column_name
            if column_name in self.data_frame.columns:
                del self.data_frame[column_name]
            self.data_frame = self.data_frame.join(series)
            self.data_frame[column_name] = self.data_frame[column_name].fillna(False)
        elif cond2:
            slct_dates = self.day_index[boolean_index].date
            bix = np.isin(self.data_frame.index.date, slct_dates)
            self.data_frame[column_name] = False
            self.data_frame.loc[bix, column_name] = True
        if column_name in self.data_frame_raw.columns:
            del self.data_frame_raw[column_name]
        temp = (self.data_frame[[column_name, self.seq_index_key]]).copy()
        temp = temp.dropna()
        temp = temp.set_index(self.seq_index_key)
        self.data_frame_raw = self.data_frame_raw.join(
            temp, on=self.seq_index_key
        )

    def fix_dst(self):
        """
        Helper function for fixing data sets with known DST shift. This function
        works for data recorded anywhere in the United States. The choice of
        timezone (e.g. 'US/Pacific') does not matter, as long as the dates
        of the clock changes are the same.
        :return:
        """
        if not self.__fix_dst_ran:
            df = self.data_frame_raw
            df_localized = df.tz_localize('US/Pacific', ambiguous='NaT',
                                          nonexistent='NaT')
            df_localized = df_localized[df_localized.index == df_localized.index]
            df_localized = df_localized.tz_convert('Etc/GMT+8')
            df_localized = df_localized.tz_localize(None)
            self.data_frame_raw = df_localized
            self.__fix_dst_ran = True
            return
        else:
            print('DST correction already performed on this data set.')
            return

    def make_data_matrix(self, use_col=None, start_day_ix=None, end_day_ix=None):
        df = self.data_frame
        if use_col is None:
            use_col = df.columns[0]
        self.raw_data_matrix, day_index = make_2d(df, key=use_col, return_day_axis=True)
        self.raw_data_matrix = self.raw_data_matrix[:, start_day_ix:end_day_ix]
        self.num_days = self.raw_data_matrix.shape[1]
        if self.raw_data_matrix.shape[0] <= 1400:
            self.data_sampling = int(24 * 60 / self.raw_data_matrix.shape[0])
        else:
            self.data_sampling = 24 * 60 / self.raw_data_matrix.shape[0]
        self.use_column = use_col
        self.day_index = day_index[start_day_ix:end_day_ix]
        d1 = self.day_index[0].strftime('%x')
        d2 = self.day_index[-1].strftime('%x')
        self.data_frame = self.data_frame[d1:d2]
        self.start_doy = self.day_index.dayofyear[0]
        return

    def make_filled_data_matrix(self, zero_night=True, interp_day=True):
        self.filled_data_matrix = np.copy(self.raw_data_matrix)
        if zero_night:
            self.filled_data_matrix = zero_nighttime(self.raw_data_matrix,
                                                     night_mask=~self.boolean_masks.daytime)
        if interp_day:
            self.filled_data_matrix = interp_missing(self.filled_data_matrix)
        else:
            msk = np.isnan(self.filled_data_matrix)
            self.filled_data_matrix[msk] = 0
        self.daily_signals.energy = np.sum(self.filled_data_matrix, axis=0) *\
                                   24 / self.filled_data_matrix.shape[1]
        return

    def generate_extra_matrix(self, column, new_index=None, key=None):
        if new_index is None:
            freq = self.data_sampling * 60
            end = self.day_index[-1].date() + timedelta(days=1)
            new_index = pd.date_range(start=self.day_index[0].date(),
                                      end=end,
                                      freq='{}s'.format(freq))[:-1]
        num_meas = self.filled_data_matrix.shape[0]
        new_view = self.data_frame[column].loc[new_index[0]:new_index[-1]]
        new_view = new_view.values.reshape(num_meas, -1, order='F')
        if self.time_shifts:
            ts = self.time_shift_analysis
            new_view = ts.apply_corrections(new_view)
        if key is None:
            key = column
        self.extra_matrices[key] = new_view
        self.extra_quality_scores[key] = (
            1 - np.sum(np.isnan(new_view[self.boolean_masks.daytime]))
            / np.sum(self.boolean_masks.daytime)
        )
        return

    def get_daily_scores(self, threshold=0.2, solver=None):
        self.get_density_scores(threshold=threshold, solver=solver) # CVXPY
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
        self.__density_lower_threshold = density_lower_threshold
        self.__density_upper_threshold = density_upper_threshold
        self.__linearity_threshold = linearity_threshold
        self.daily_scores.quality_clustering = db.labels_

    def get_density_scores(self, threshold=0.2, solver=None):
        if self.raw_data_matrix is None:
            print('Generate a raw data matrix first.')
            return
        s1, s2, s3 = daily_missing_data_advanced(
            self.raw_data_matrix, threshold=threshold,
            return_density_signal=True, return_fit=True, solver=solver
        )
        self.daily_scores.density = s1
        self.daily_signals.density = s2
        self.daily_signals.seasonal_density_fit = s3
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
        # Label detected infill points with a boolean mask
        infill = np.zeros_like(self.raw_data_matrix, dtype=np.bool)
        slct = self.daily_scores.linearity >= 0.1
        reference_diffs = np.tile(modes[0][slct],
                                  (self.filled_data_matrix.shape[0], 1))
        found_infill = np.logical_or(
            np.isclose(
                np.r_[np.zeros(self.num_days).reshape((1, -1)),
                      difference_mat][ :, slct],
                reference_diffs),
            np.isclose(
                np.r_[difference_mat,
                      np.zeros(self.num_days).reshape((1, -1))][:, slct],
                reference_diffs),
        )
        infill[:, slct] = found_infill
        self.boolean_masks.infill = infill
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
        return

    def clipping_check(self, solver=None):
        if self.clipping_analysis is None:
            self.clipping_analysis = ClippingDetection()
        self.clipping_analysis.check_clipping(
            self.filled_data_matrix, no_error_flag=self.daily_flags.no_errors,
            solver=solver
        )
        self.inverter_clipping = self.clipping_analysis.inverter_clipping
        self.num_clip_points = self.clipping_analysis.num_clip_points
        self.daily_scores.clipping_1 = self.clipping_analysis.clip_stat_1
        self.daily_scores.clipping_2 = self.clipping_analysis.clip_stat_2
        self.daily_flags.inverter_clipped = self.clipping_analysis.clipped_days

    def find_clipped_times(self):
        if self.clipping_analysis is None:
            self.clipping_check()
        self.clipping_analysis.find_clipped_times()
        self.boolean_masks.clipped_times = self.clipping_analysis.clipping_mask

    def capacity_clustering(self, solver=None, plot=False, figsize=(8, 6),
                            show_clusters=True):
        if self.capacity_analysis is None:
            self.capacity_analysis = CapacityChange()
            self.capacity_analysis.run(
                self.filled_data_matrix, filter=self.daily_flags.no_errors,
                quantile=1.00, c1=15, c2=100, c3=300, reweight_eps=0.5,
                reweight_niter=5, dbscan_eps=.02, dbscan_min_samples='auto',
                solver=solver
            )
        if len(set(self.capacity_analysis.labels)) > 1: #np.max(db.labels_) > 0:
            self.capacity_changes = True
            self.daily_flags.capacity_cluster = self.capacity_analysis.labels
        else:
            self.capacity_changes = False
        if plot:
            metric = self.capacity_analysis.metric
            s1 = self.capacity_analysis.s1
            s2 = self.capacity_analysis.s2
            labels = self.capacity_analysis.labels
            try:
                xs = self.day_index.to_pydatetime()
            except AttributeError:
                xs = np.arange(self.num_days)
            if show_clusters:
                fig, ax = plt.subplots(nrows=2, figsize=figsize, sharex=True,
                                       gridspec_kw={'height_ratios': [4, 1]})
                ax[0].plot(xs, s1, label='capacity change detector')
                ax[0].plot(xs, s2 + s1, label='signal model')
                ax[0].plot(xs, metric, alpha=0.3,
                           label='measured signal')
                ax[0].legend()
                ax[0].set_title('Detection of system capacity changes')
                ax[1].set_xlabel('date')
                ax[0].set_ylabel('normalized daily max power')
                ax[1].plot(xs, labels, ls='none', marker='.')
                ax[1].set_ylabel('Capacity clusters')
            else:
                fig, ax = plt.subplots(nrows=1, figsize=figsize)
                ax.plot(xs, s1, label='capacity change detector')
                ax.plot(xs, s2 + s1, label='signal model')
                ax.plot(xs, metric, alpha=0.3,
                         label='measured signal')
                ax.legend()
                ax.set_title('Detection of system capacity changes')
                ax.set_ylabel('normalized daily maximum power')
                ax.set_xlabel('date')
            return fig


    def auto_fix_time_shifts(self, c1=5., c2=500., estimator='com',
                             threshold=0.1, periodic_detector=False,
                             solver=None):
        self.time_shift_analysis = TimeShift()
        use_ixs = self.daily_flags.clear
        self.time_shift_analysis.run(
            self.filled_data_matrix, use_ixs=use_ixs,
            c1=c1, c2=c2, solar_noon_estimator=estimator, threshold=threshold,
            periodic_detector=periodic_detector, solver=solver
        )
        self.filled_data_matrix = self.time_shift_analysis.corrected_data
        if len(self.time_shift_analysis.index_set) == 0:
            self.time_shifts = False
        else:
            self.time_shifts = True

    def detect_clear_days(self, smoothness_threshold=0.9, energy_threshold=0.8,
                          solver=None):
        if self.filled_data_matrix is None:
            print('Generate a filled data matrix first.')
            return
        clear_days = find_clear_days(self.filled_data_matrix,
                                     smoothness_threshold=smoothness_threshold,
                                     energy_threshold=energy_threshold,
                                     solver=solver)
        ### Remove days that are marginally low density, but otherwise pass
        # the clearness test. Occasionally, we find an early morning or late
        # afternoon inverter outage on a clear day is still detected as clear.
        # Added July 2020 --BM
        clear_days = np.logical_and(
            clear_days,
            self.daily_scores.density > 0.9
        )
        self.daily_flags.flag_clear_cloudy(clear_days)
        return

    def find_clear_times(self, power_hyperparam=0.1,
                         smoothness_hyperparam=0.05, min_length=3):
        if self.scsf is None:
            print('No SCSF model detected. Fitting now...')
            self.fit_statistical_clear_sky_model()
        clear = self.scsf.estimated_power_matrix
        clear_times = find_clear_times(self.filled_data_matrix, clear,
                                       self.capacity_estimate,
                                       th_relative_power=power_hyperparam,
                                       th_relative_smoothness=smoothness_hyperparam,
                                       min_length=min_length)
        self.boolean_masks.clear_times = clear_times


    def fit_statistical_clear_sky_model(self, rank=6, mu_l=None, mu_r=None,
                                        tau=None, exit_criterion_epsilon=1e-3,
                                        solver_type='MOSEK', max_iteration=10,
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
        scsf = SCSF(data_handler_obj=self, rank_k=rank, solver_type=solver_type)
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

    def calculate_scsf_performance_index(self):
        if self.scsf is None:
            print('No SCSF model detected. Fitting now...')
            self.fit_statistical_clear_sky_model()
        clear = self.scsf.estimated_power_matrix
        clear_energy = np.sum(clear, axis=0)
        measured_energy = np.sum(self.filled_data_matrix, axis=0)
        pi = np.divide(measured_energy, clear_energy)
        return pi


    def plot_heatmap(self, matrix='raw', flag=None, figsize=(12, 6),
                     scale_to_kw=True, year_lines=True, units=None):
        if matrix == 'raw':
            mat = np.copy(self.raw_data_matrix)
        elif matrix == 'filled':
            mat = np.copy(self.filled_data_matrix)
        elif matrix in self.extra_matrices.keys():
            mat = self.extra_matrices[matrix]
        else:
            return
        if units is None:
            if scale_to_kw and self.power_units == 'W':
                mat /= 1000
                units = 'kW'
            else:
                units = self.power_units
        if flag is None:
            return plot_2d(mat, figsize=figsize,
                           dates=self.day_index, year_lines=year_lines,
                           units=units)
        elif flag == 'good':
            fig = plot_2d(mat, figsize=figsize,
                          clear_days=self.daily_flags.no_errors,
                          dates=self.day_index, year_lines=year_lines,
                          units=units)
            plt.title('Measured power, good days flagged')
            return fig
        elif flag == 'bad':
            fig = plot_2d(mat, figsize=figsize,
                          clear_days=~self.daily_flags.no_errors,
                          dates=self.day_index, year_lines=year_lines,
                          units=units)
            plt.title('Measured power, bad days flagged')
            return fig
        elif flag in ['clear', 'sunny']:
            fig = plot_2d(mat, figsize=figsize,
                          clear_days=self.daily_flags.clear,
                          dates=self.day_index, year_lines=year_lines,
                          units=units)
            plt.title('Measured power, clear days flagged')
            return fig
        elif flag == 'cloudy':
            fig = plot_2d(mat, figsize=figsize,
                          clear_days=self.daily_flags.cloudy,
                          dates=self.day_index, year_lines=year_lines,
                          units=units)
            plt.title('Measured power, cloudy days flagged')
            return fig
        elif flag == 'clipping':
            fig = plot_2d(mat, figsize=figsize,
                          clear_days=self.daily_flags.inverter_clipped,
                          dates=self.day_index, year_lines=year_lines,
                          units=units)
            plt.title('Measured power, days with inverter clipping flagged')
            return fig
        else:
            print('Unknown daily flag. Please use one of the following:')
            print('good, bad, sunny, cloudy, clipping')
            return

    def plot_daily_signals(self, boolean_index=None, start_day=0, num_days=5,
                           filled=True, ravel=True, figsize=(12, 6),
                           color=None, alpha=None, label=None,
                           boolean_mask=None, mask_label=None,
                           show_clear_model=True, show_legend=False,
                           marker=None):
        if type(start_day) is not int:
            try:
                loc = self.day_index == start_day
                start_day = np.arange(self.num_days)[loc][0]
            except IndexError:
                print("Please use an integer or a date string for 'start_day'")
                return
        if boolean_index is None:
            boolean_index = np.s_[:]
        i = start_day
        j = start_day + num_days
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
        if marker is not None:
            kwargs['marker'] = marker
        if self.day_index is not None:
            start = self.day_index[start_day]
            freq = '{}min'.format(self.data_sampling)
            periods = len(plot_data)
            xs = pd.date_range(start=start, freq=freq, periods=periods)
        else:
            xs = np.arange(len(plot_data))
        if label is None:
            label = 'measured power'
        plt.plot(xs, plot_data, linewidth=1, **kwargs, label=label)
        if boolean_mask is not None:
            if mask_label is None:
                mask_label = 'boolean mask'
            m, n = self.raw_data_matrix.shape
            index_shape = boolean_mask.shape
            cond1 = index_shape == (m, n)
            cond2 = index_shape == (n,)
            if cond1:
                plot_flags = boolean_mask[:, slct].ravel(order='F')
            elif cond2:
                temp_bool = np.tile(boolean_mask, (m, 1))
                plot_flags = temp_bool[:, slct].ravel(order='F')
            plt.plot(xs[plot_flags], plot_data[plot_flags], ls='none',
                     marker='.', color='red', label=mask_label)
        if show_clear_model and self.scsf is not None:
            plot_model = self.scsf.estimated_power_matrix[:, slct].ravel(order='F')
            plt.plot(xs, plot_model, color='orange', linewidth=1,
                     label='clear sky model')
        if show_legend:
            plt.legend()
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
        if self.daily_scores is None:
            return
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
        return self.clipping_analysis.plot_pdf(figsize=figsize)

    def plot_daily_max_cdf(self, figsize=(10, 6)):
        return self.clipping_analysis.plot_cdf(figsize=figsize)

    def plot_daily_max_cdf_and_pdf(self, figsize=(10, 6)):
        return self.clipping_analysis.plot_both(figsize=figsize)

    def plot_cdf_analysis(self, figsize=(12, 6)):
        return self.clipping_analysis.plot_diffs(figsize=figsize)

    def plot_capacity_change_analysis(self, figsize=(8, 6), show_clusters=True):
        fig = self.capacity_clustering(plot=True, figsize=figsize,
                                       show_clusters=show_clusters)
        return fig

    def plot_time_shift_analysis_results(self, figsize=(8, 6)):
        if self.time_shift_analysis is not None:
            use_ixs = self.time_shift_analysis.use_ixs
            plt.figure(figsize=figsize)
            plt.plot(self.day_index, self.time_shift_analysis.metric,
                     linewidth=1, alpha=0.6,
                     label='daily solar noon')
            plt.plot(self.day_index[use_ixs],
                     self.time_shift_analysis.metric[use_ixs],
                     linewidth=1, alpha=0.6, color='orange', marker='.',
                     ls='none',
                     label='filtered days')
            plt.plot(self.day_index, self.time_shift_analysis.s1, color='green',
                     label='shift detector')
            plt.plot(self.day_index,
                     self.time_shift_analysis.s1 + self.time_shift_analysis.s2,
                     color='red', label='signal model', ls='--')
            # plt.ylim(11, 13)
            plt.legend()
            fig = plt.gcf()
            return fig
        else:
            print('Please run pipeline first.')

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
        self.capacity_cluster = None

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

class BooleanMasks():
    """
    Boolean masks are used to identify time periods corresponding to elements
    in the data matrix. The masks have the same shape as the data matrix. The
    masks can be used to select data according to certain rules, generate the
    associated time stamp values, or perform other data maniuplation. See,
    for example:

        https://jakevdp.github.io/PythonDataScienceHandbook/02.06-boolean-arrays-and-masks.html
    """
    def __init__(self):
        self.clear_times = None
        self.clipped_times = None
        self.daytime = None
        self.missing_values = None
        self.infill = None