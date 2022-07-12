""" Latitude Study Module
This module contains a class for conducting a study to estimate latitude from solar data. This code accepts solar data
in the form of a `solar-data-tools` `DataHandler` object, which is used to standardize and pre-process the data. The
provided class will then estimate the latitude of the site that produced the data, using the `run` method.

The following configurations can be run:

 - Input data matrix: 'raw', 'filled'
 - Daylight estimation method: 'raw daylight', 'sunrise-sunset', 'optimized_estimates', 'optimized_measurements'
 - Declination equation: 'cooper', 'spencer'
 - Day selection method: 'all', 'clear', 'cloudy'

"""
import numpy as np
import pandas as pd
from pvsystemprofiler.utilities.declination_equation import delta_spencer
from pvsystemprofiler.utilities.declination_equation import delta_cooper
from pvsystemprofiler.algorithms.latitude.hours_daylight import calculate_hours_daylight
from pvsystemprofiler.algorithms.latitude.hours_daylight import calculate_hours_daylight_raw
from pvsystemprofiler.algorithms.optimized_sunrise_sunset import get_optimized_sunrise_sunset
from pvsystemprofiler.algorithms.latitude.estimation import estimate_latitude


class LatitudeStudy():
    def __init__(self, data_handler, lat_true_value=None):
        """
        :param data_handler: `DataHandler` class instance loaded with a solar power data set.
        :param lat_true_value: Optional. The ground truth value for the system's latitude. (Degrees).
        """

        self.data_handler = data_handler
        self.latitude_true_value = lat_true_value
        if not data_handler._ran_pipeline:
            print('Running DataHandler preprocessing pipeline with defaults')
            self.data_handler.run_pipeline()
        self.data_matrix = self.data_handler.filled_data_matrix
        self.raw_data_matrix = self.data_handler.raw_data_matrix
        self.day_of_year = self.data_handler.day_index.dayofyear
        self.num_days = self.data_handler.num_days
        self.daily_meas = self.data_handler.filled_data_matrix.shape[0]
        self.data_sampling = self.data_handler.data_sampling
        self.boolean_daytime = None
        self.hours_daylight = None
        self.delta = None
        self.delta_cooper = None
        self.delta_spencer = None
        self.residual = None
        self.daytime_threshold = None
        self.opt_threshold = None
        self.days = None
        self.estimates_sunrise_raw = None
        self.estimates_sunset_raw = None
        self.measurements_sunrise_raw = None
        self.measurements_sunset_raw = None
        self.estimates_sunrise_filled = None
        self.estimates_sunset_filled = None
        self.measurements_sunrise_filled = None
        self.measurements_sunset_filled = None
        self.opt_threshold_filled = None
        self.opt_threshold_raw = None
        # Results
        self.results = None

    def run(self, data_matrix=('raw', 'filled'),
            daylight_method=('raw daylight', 'sunrise-sunset', 'optimized_estimates', 'optimized_measurements'),
            delta_method=('cooper', 'spencer'), day_selection_method=('all', 'clear', 'cloudy'),
            threshold=None):
        """
        Run a study with the given configuration of options. Defaults to
        running all available options. Any kwarg can be constrained by
        providing a subset of acceptable keys. For example the default keys
        for daylight method kwarg are:

            ('raw daylight', 'sunrise-sunset', 'optimized_estimates', 'optimized_measurements')

        Additionally, any of the following would be acceptable for this kwarg:

            ('raw daylight', 'sunrise-sunset', 'optimized_estimates', 'optimized_measurements')
            ('raw daylight', 'sunrise-sunset', 'optimized_estimates')
            ('raw daylight', 'sunrise-sunset')
            ('raw daylight')

        This method sets the `results` attribute to be a pandas data frame
        containing the results of the study.
        :param data_matrix: 'raw', 'filled'.
        :param daylight_method: 'raw daylight', 'sunrise-sunset', 'optimized_estimates', 'optimized_measurements'.
        :param threshold: (optional) daylight threshold values, tuple of length one to twelve.
        :param delta_method: (optional) 'cooper', 'spencer'.
        :param day_selection_method: 'all', 'clear', 'cloudy'.
        :return: None.
        """
        data_matrix = np.atleast_1d(data_matrix)
        daylight_method = np.atleast_1d(daylight_method)
        delta_method = np.atleast_1d(delta_method)
        day_selection_method = np.atleast_1d(day_selection_method)

        if threshold is None:
            self.daytime_threshold = 0.001 * np.ones(len(data_matrix) * len(daylight_method) * len(delta_method) *
                                                     len(day_selection_method))
        else:
            self.daytime_threshold = threshold

        self.delta_cooper = delta_cooper(self.day_of_year, self.daily_meas)
        self.delta_spencer = delta_spencer(self.day_of_year, self.daily_meas)

        if 'raw' in data_matrix:
            rdm = self.raw_data_matrix
        else:
            rdm = None
        if 'filled' in data_matrix:
            fdm = self.data_matrix
        else:
            fdm = None

        opt_dict = get_optimized_sunrise_sunset(fdm, rdm)
        self.estimates_sunrise_raw, self.estimates_sunset_raw, self.measurements_sunrise_raw, \
        self.measurements_sunset_raw, self.opt_threshold_raw, \
        self.estimates_sunrise_filled, self.estimates_sunset_filled, self.measurements_sunrise_filled, \
        self.measurements_sunset_filled, self.opt_threshold_filled = opt_dict.values()

        results = pd.DataFrame(columns=['declination_method', 'daylight_calculation', 'data_matrix', 'threshold',
                                        'day_selection_method', 'latitude'])
        counter = 0
        for delta_id in delta_method:
            for matrix_ix, matrix_id in enumerate(data_matrix):

                for daylight_method_id in daylight_method:
                    if daylight_method_id != 'optimized_estimates':
                        dtt = self.daytime_threshold[counter]
                    else:
                        dtt = None

                    for ds in day_selection_method:
                        if ds == 'all':
                            self.days = self.data_handler.daily_flags.no_errors
                        elif ds == 'clear':
                            self.days = self.data_handler.daily_flags.clear
                        elif ds == 'cloudy':
                            self.days = self.data_handler.daily_flags.cloudy
                        dlm = daylight_method_id
                        tm = data_matrix[matrix_ix]
                        dm = delta_id

                        self.prepare_input_data(matrix_id, daytime_threshold=dtt, daylight_method=dlm,
                                                delta_method=delta_id)
                        lat_est = estimate_latitude(self.hours_daylight, self.delta)

                        if daylight_method_id in ['optimized_estimates', 'optimized_measurements']:
                            dtt = self.opt_threshold

                        results.loc[counter] = [dm, dlm, tm, dtt, ds, lat_est]
                        counter += 1
        if self.latitude_true_value is not None:
            results['residual'] = self.latitude_true_value - results['latitude']
            results['measured_latitude'] = self.latitude_true_value

        self.results = results

    def prepare_input_data(self, matrix_id=None, daytime_threshold=0.001,
                           daylight_method=('sunrise-sunset', 'raw daylight'),
                           delta_method=('cooper', 'spencer')):
        """"
        Latitude is estimated from equation (1.6.11) in:
        Duffie, John A., and William A. Beckman. Solar engineering of thermal processes. New York: Wiley, 1991.
        """

        if matrix_id == 'raw':
            data_in = self.raw_data_matrix
        elif matrix_id == 'filled':
            data_in = self.data_matrix
        if daylight_method in ('sunrise-sunset', 'sunrise sunset'):
            hours_daylight_all = calculate_hours_daylight(data_in, daytime_threshold)
        elif daylight_method in ('raw_daylight', 'raw daylight'):
            hours_daylight_all = calculate_hours_daylight_raw(data_in, self.data_sampling, daytime_threshold)
        elif daylight_method in ('optimized_estimates', 'Optimized_Estimates'):
            if matrix_id == 'filled':
                hours_daylight_all = self.estimates_sunset_filled - self.estimates_sunrise_filled
                self.opt_threshold = self.opt_threshold_raw
            if matrix_id == 'raw':
                hours_daylight_all = self.estimates_sunset_raw - self.estimates_sunrise_raw
                self.opt_threshold = self.opt_threshold_filled
        elif daylight_method in ('optimized_measurements', 'Optimized_Measurements'):
            if matrix_id == 'filled':
                hours_daylight_all = self.measurements_sunset_filled - self.measurements_sunrise_filled
                self.opt_threshold = self.opt_threshold_filled
            if matrix_id == 'raw':
                hours_daylight_all = self.measurements_sunset_raw - self.measurements_sunrise_raw
                self.opt_threshold = self.opt_threshold_raw
        if delta_method in ('Cooper', 'cooper'):
            self.delta = self.delta_cooper
        elif delta_method in ('Spencer', 'spencer'):
            self.delta = self.delta_spencer

        if np.any(np.isnan(hours_daylight_all)):
            hours_mask = np.isnan(hours_daylight_all)
            full_mask = ~hours_mask & self.days
            self.hours_daylight = hours_daylight_all[full_mask]
            self.delta = self.delta[:, full_mask]
        else:
            self.hours_daylight = hours_daylight_all[self.days]
            self.delta = self.delta[:, self.days]
        return
