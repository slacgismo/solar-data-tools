"""
This module contains a class for estimating longitude, latitude, tilt and azimuth from an input signal. This code
accepts an input signal data in the form of a `solar-data-tools` `DataHandler` object, which is used to standardize
and pre-process the data. The provided class will then estimate  longitude, latitude, tilt and azimuth using the
 'estimate_longitude', 'estimate_latitude' and 'estimate_orientation' methods, respectively. Alternatively, all four
 parameters can be estimated at once using the 'estimate_all' method.
"""
# Standard Imports
import numpy as np

# Solar Data Tools Imports
from solardatatools.solar_noon import energy_com, avg_sunrise_sunset

# Module Imports
from pvsystemprofiler.utilities.equation_of_time import eot_da_rosa, eot_duffie
from solardatatools.algorithms import SunriseSunset
from pvsystemprofiler.algorithms.latitude.hours_daylight import calculate_hours_daylight
from pvsystemprofiler.utilities.hour_angle_equation import calculate_omega
from pvsystemprofiler.utilities.declination_equation import delta_cooper
from pvsystemprofiler.algorithms.angle_of_incidence.curve_fitting import run_curve_fit
from pvsystemprofiler.algorithms.performance_model_estimation import find_fit_costheta
from pvsystemprofiler.algorithms.angle_of_incidence.lambda_functions import (
    select_function,
)
from pvsystemprofiler.algorithms.angle_of_incidence.dynamic_value_functions import (
    determine_keys,
)
from pvsystemprofiler.algorithms.angle_of_incidence.dynamic_value_functions import (
    select_init_values,
)
from pvsystemprofiler.algorithms.tilt_azimuth.daytime_threshold_quantile import (
    filter_data,
)
from pvsystemprofiler.utilities.tools import random_initial_values
from pvsystemprofiler.algorithms.longitude.estimation import estimate_longitude
from pvsystemprofiler.algorithms.latitude.estimation import estimate_latitude


class ConfigurationEstimator:
    def __init__(
        self,
        data_handler,
        gmt_offset,
        solar_noon_method="optimized_estimates",
        daylight_method="optimized_estimates",
        data_matrix="filled",
        daytime_threshold=None,
    ):
        if not data_handler._ran_pipeline:
            data_handler.run_pipeline()
        self.data_handler = data_handler
        if data_matrix == "raw":
            self.data_matrix = data_handler.raw_data_matrix
        elif data_matrix == "filled":
            self.data_matrix = data_handler.filled_data_matrix
        # Parameters to be estimated
        self.longitude = None
        self.latitude = None
        self.tilt = None
        self.azimuth = None
        # Attributes used for all calculations
        self.gmt_offset = gmt_offset
        self.hours_daylight = None
        self.daily_meas = self.data_handler.filled_data_matrix.shape[0]
        self.daytime_threshold = None
        self.day_interval = None
        self.x1 = None
        self.x2 = None
        self.data_sampling = self.data_handler.data_sampling
        self.num_days = self.data_handler.num_days
        self.day_of_year = self.data_handler.day_index.dayofyear
        self.eot_duffie = eot_duffie(self.day_of_year)
        self.eot_da_rosa = eot_da_rosa(self.day_of_year)
        self.delta = delta_cooper(self.day_of_year, self.daily_meas)
        self.omega = None
        self.days = np.logical_and(
            data_handler.daily_flags.clear, ~data_handler.daily_flags.inverter_clipped
        )
        # self.days = self.data_handler.daily_flags.no_errors

        if (
            solar_noon_method == "optimized_estimates"
            or daylight_method == "optimized_estimates"
        ):
            ss = self.data_handler.daytime_analysis

        if solar_noon_method == "rise_set_average":
            self.solarnoon = avg_sunrise_sunset(self.data_matrix)
        elif solar_noon_method == "energy_com":
            self.solarnoon = energy_com(self.data_matrix)
        if solar_noon_method == "optimized_estimates":
            self.solarnoon = np.nanmean(
                [ss.sunrise_estimates, ss.sunset_estimates], axis=0
            )

        if daylight_method in ("sunrise-sunset", "sunrise sunset"):
            self.hours_daylight = calculate_hours_daylight(
                self.data_matrix, daytime_threshold
            )
        elif daylight_method == "optimized_estimates":
            self.hours_daylight = ss.sunset_estimates - ss.sunrise_estimates

    def estimate_longitude(self, estimator="fit_l1", eot_calculation="duffie"):
        """
        :param estimator: 'calculated', 'fit_l1', 'fit_l2' or 'fit_huber'
        :param eot_calculation: 'rise_set_average', 'energy_com', 'optimized_estimates',
        or 'optimized_measurements'
        :return: None
        """
        if eot_calculation in ("duffie", "d", "duf"):
            eot = self.eot_duffie
        elif eot_calculation in ("da_rosa", "dr", "rosa"):
            eot = self.eot_da_rosa
        self.longitude = estimate_longitude(
            estimator, eot, self.solarnoon, self.days, self.gmt_offset
        )
        return

    def estimate_latitude(self):
        hours_daylight_filtered, delta_filtered = self._prepare_lat_input_data()
        self.latitude = estimate_latitude(hours_daylight_filtered, delta_filtered)
        return

    def _prepare_lat_input_data(self):
        if np.any(np.isnan(self.hours_daylight)):
            hours_mask = np.isnan(self.hours_daylight)
            full_mask = ~hours_mask & self.days
            hours_daylight_filtered = self.hours_daylight[full_mask]
            delta = self.delta[:, full_mask]
        else:
            hours_daylight_filtered = self.hours_daylight[self.days]
            delta = self.delta[:, self.days]
        return hours_daylight_filtered, delta

    # estimate tilt and azimuth with or without longitude and latitude input values
    def estimate_orientation(
        self,
        longitude=None,
        latitude=None,
        tilt=None,
        azimuth=None,
        day_interval=None,
        x1=0.9,
        x2=0.9,
    ):
        """
        Estimates tilt and azimuth. The intended use is to estimate tilt and azimuth given longitude and latitude.
        However, the algorithm will estimate any of longitude, latitude, tilt and azimuth depending on the input values.
        If a parameter is not provided as an input value, it will be estimated. Any other use than the stated above as
        the intended use was found to yield inaccurate results.

        :param longitude: optional. Longitude value to be used in parameter estimation.
        :param latitude: optional. Latitude value to be used in parameter estimation.
        :param tilt: optional. Tilt value to be used in parameter estimation.
        :param azimuth: optional. Azimuth value to be used in parameter estimation.
        :param day_interval: 'all', 'clear' or 'cloudy'.
        :param x1: cvx parameter. Factor used in signal decomposition for estimation of daytime threshold.
        :param x2: Quantile of data used in estimation of daytime threshold.
        :return: None
        """

        if longitude is None:
            est_lon = ConfigurationEstimator(self.data_handler, self.gmt_offset)
            est_lon.estimate_longitude()
            self.longitude = est_lon.longitude
        else:
            self.longitude = longitude
        if latitude is None:
            est_lat = ConfigurationEstimator(self.data_handler, self.gmt_offset)
            est_lat.estimate_latitude()
            self.latitude = est_lat.latitude
        else:
            self.latitude = latitude

        self.tilt = tilt
        self.azimuth = azimuth
        self.day_interval = day_interval
        self.x1 = x1
        self.x2 = x2
        dh = self.data_handler
        self.data_matrix = dh.filled_data_matrix
        self.num_days = dh.num_days
        self.omega = calculate_omega(
            self.day_of_year, self.data_sampling, self.longitude, self.gmt_offset
        )

        self.tilt, self.azimuth = self._cal_orientation_helper()

    def estimate_all(self, day_interval=None, x1=0.9, x2=0.9):
        """
        Estimate latitude, longitude, tilt and azimuth all at once.
        :param day_interval: 'all', 'clear', 'cloudy'.
        :param x1: cvx parameter. Factor used in signal decomposition for estimation of daytime threshold.
        :param x2: Quantile of data used in estimation of daytime threshold.
        """

        self.tilt = None
        self.azimuth = None
        self.day_interval = day_interval
        self.x1 = x1
        self.x2 = x2
        dh = self.data_handler
        self.data_matrix = dh.filled_data_matrix
        self.days = dh.daily_flags.clear
        self.num_days = dh.num_days
        self.delta = delta_cooper(self.day_of_year, self.daily_meas)
        est_lon = ConfigurationEstimator(self.data_handler, self.gmt_offset)
        est_lon.estimate_longitude()
        self.longitude = est_lon.longitude
        est_lat = ConfigurationEstimator(self.data_handler, self.gmt_offset)
        est_lat.estimate_latitude()
        self.latitude = est_lat.latitude
        self.omega = calculate_omega(
            self.day_of_year, self.data_sampling, self.longitude, self.gmt_offset
        )

        self.tilt, self.azimuth = self._cal_orientation_helper()

    def _cal_orientation_helper(self):
        if self.day_interval is not None:
            day_range = (self.day_of_year > self.day_interval[0]) & (
                self.day_of_year < self.day_interval[1]
            )
        else:
            day_range = np.ones(self.day_of_year.shape, dtype=bool)

        doy = self.data_handler.day_index.day_of_year
        normalized_data, costheta_est = find_fit_costheta(
            self.data_matrix, self.days, doy
        )

        self.data_handler.find_clipped_times()
        boolean_filter = normalized_data >= 0.15 * np.exp(costheta_est)

        boolean_filter = (
            boolean_filter
            * ~self.data_handler.boolean_masks.clipped_times
            * self.data_handler.daily_flags.clear
            * day_range
        )

        delta_f = self.delta[boolean_filter]
        omega_f = self.omega[boolean_filter]
        if ~np.any(boolean_filter):
            print("No data made it through filters")

        lat_initial, tilt_initial, azim_initial = random_initial_values(1)

        func_customized, bounds = select_function(
            self.latitude, self.tilt, self.azimuth
        )
        dict_keys = determine_keys(
            latitude=self.latitude, tilt=self.tilt, azimuth=self.azimuth
        )

        init_values_dict = {
            "latitude": lat_initial[0],
            "tilt": tilt_initial[0],
            "azimuth": azim_initial[0],
        }
        init_values, ivr = select_init_values(init_values_dict, dict_keys)

        estimates = run_curve_fit(
            func=func_customized,
            keys=dict_keys,
            delta=delta_f,
            omega=omega_f,
            costheta=normalized_data,
            boolean_filter=boolean_filter,
            init_values=init_values,
            fit_bounds=bounds,
        )

        for i, estimate in enumerate(dict_keys):
            if estimate == "latitude_estimate":
                lat_estimate = estimates[i]
            if estimate == "tilt_estimate":
                tilt_estimate = estimates[i]
            if estimate == "azimuth_estimate":
                azimuth_estimate = estimates[i]

        if "tilt_estimate" not in dict_keys:
            tilt_estimate = None
        if "azimuth_estimate" not in dict_keys:
            azimuth_estimate = None
        return tilt_estimate, azimuth_estimate
