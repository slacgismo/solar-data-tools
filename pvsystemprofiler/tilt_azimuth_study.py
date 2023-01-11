""" Tilt and Azimuth Study Module
This module contains a class for conducting a study to estimating Tilt and Azimuth from an input signal. This code
accepts an input signal data in the form of a `solar-data-tools` `DataHandler` object, which is used to standardize
and pre-process the data. The provided class will then estimate the Tilt and Azimuth of the site that produced the data,
using the `run` method. Tilt and Azimuth are estimated via numerical fit using equation (1.6.2) in:
Duffie, John A., and William A. Beckman. Solar engineering of thermal processes. New York: Wiley, 1991.

The following configurations can be run:
 - Day range: 'full_year' or customized day range
 - Declination equation: 'cooper', 'spencer'.
 """
import numpy as np
import pandas as pd
from pvsystemprofiler.utilities.hour_angle_equation import calculate_omega
from pvsystemprofiler.utilities.declination_equation import delta_spencer
from pvsystemprofiler.utilities.declination_equation import delta_cooper
from pvsystemprofiler.algorithms.angle_of_incidence.curve_fitting import run_curve_fit
from pvsystemprofiler.algorithms.angle_of_incidence.calculation import (
    calculate_costheta,
)
from pvsystemprofiler.algorithms.performance_model_estimation import find_fit_costheta
from pvsystemprofiler.algorithms.angle_of_incidence.lambda_functions import (
    select_function,
)
from pvsystemprofiler.utilities.angle_of_incidence_function import func_costheta
from pvsystemprofiler.algorithms.angle_of_incidence.dynamic_value_functions import (
    determine_keys,
)
from pvsystemprofiler.algorithms.angle_of_incidence.dynamic_value_functions import (
    select_init_values,
)
from pvsystemprofiler.utilities.tools import random_initial_values
from pvsystemprofiler.algorithms.tilt_azimuth.daytime_threshold_quantile import (
    filter_data,
)


class TiltAzimuthStudy:
    def __init__(
        self,
        data_handler,
        day_range="full_year",
        init_values=None,
        nrandom_init_values=None,
        daytime_threshold=None,
        lon_input=None,
        lat_input=None,
        tilt_input=None,
        azimuth_input=None,
        lat_true_value=None,
        tilt_true_value=None,
        azimuth_true_value=None,
        gmt_offset=-8,
        cvx_parameter=None,
        threshold_quantile=None,
    ):
        """
        :param data_handler: `DataHandler` class instance loaded with a solar power data set.
        :param day_range: (optional) the desired day range to run the study. A list of the form
                              [first day, last day].
        :param init_values: (optional) Latitude, Tilt and Azimuth guess values list for numerical fit. A list of the
                form [[latitude_1,.., latitude_n], [tilt_1,.., tilt_n], [azimuth_1,.., azimuth_n]]. Default value is 10.
                (Degrees).
        :param nrandom_init_values: (optional) number of random initial values to be generated.
        :param daytime_threshold: (optional) daytime threshold
        :param lon_input: longitude estimate as obtained from the Longitude Study module in Degrees.
        :param lat_input: latitude value in Degrees.
        :param tilt_input: tilt value in Degrees.
        :param azimuth_input: azimuth value in Degrees.
        :param lat_true_value: (optional) ground truth value for the system's Latitude in Degrees.
        :param tilt_true_value: (optional) ground truth value for the system's Tilt in Degrees.
        :param azimuth_true_value: (optional) ground truth value for the system's Azimuth in Degrees Degrees.
        :param gmt_offset: The offset in hours between the local timezone and GMT/UTC.
        :param cvx_parameter: (optional). Factor used in signal decomposition for estimation of daytime threshold.
        :param threshold_quantile: (optional). Quantile of data used in estimation of daytime threshold.
        """

        self.data_handler = data_handler
        # Choose day range from season dictionary
        if day_range is None:
            self.day_range_dict = {}
            self.day_range_dict = {
                "summer": [171, 265],
                "no_winter": [79, 355],
                "spring": [79, 171],
                "winter": [355, 79],
                "winter_spring": [355, 171],
                "full_year": None,
            }

        elif day_range is "full_year":
            self.day_range_dict = {"full_year": None}
        elif day_range is "manual":
            self.day_range_dict = {"manual": day_range}

        self.data_matrix = self.data_handler.filled_data_matrix
        if not data_handler._ran_pipeline:
            print("Running DataHandler preprocessing pipeline with defaults")
            self.data_handler.run_pipeline()
        # initial values
        self.init_values = init_values
        self.nrandom = nrandom_init_values
        # inputs
        self.lon_input = lon_input
        self.lat_input = lat_input
        self.tilt_input = tilt_input
        self.azimuth_input = azimuth_input
        # true values
        self.lat_true_value = lat_true_value
        self.tilt_true_value = tilt_true_value
        self.azimuth_true_value = azimuth_true_value
        self.gmt_offset = gmt_offset
        # thresholds
        self.daytime_threshold = daytime_threshold
        self.daytime_threshold_fit = None
        if cvx_parameter is None:
            self.threshold_x1 = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        else:
            self.threshold_x1 = np.atleast_1d(cvx_parameter)
        if threshold_quantile is None:
            self.threshold_x2 = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        else:
            self.threshold_x2 = np.atleast_1d(threshold_quantile)
        # data specific variables
        self.day_of_year = self.data_handler.day_index.dayofyear
        self.num_days = self.data_handler.num_days
        self.daily_meas = self.data_handler.filled_data_matrix.shape[0]
        self.data_sampling = self.data_handler.data_sampling
        self.clear_index = data_handler.daily_flags.clear
        # angle of incidence parameters
        self.delta_cooper = None
        self.delta_spencer = None
        self.omega = None
        self.scale_factor_costheta = None
        self.costheta_estimated = None
        self.costheta_ground_truth = None
        self.costheta_fit = None
        # other
        self.results = None

    def run(self, delta_method=("cooper", "spencer")):
        """
        Run a study with the given configuration of options. Defaults to
        running all available options. Any kwarg can be constrained by
        providing a subset of acceptable keys. For example the default keys
        for the declination method estimator kwarg are:

        ('cooper', 'spencer')

        Additionally, any of the following would be acceptable for this kwarg:

        ('cooper')
        'cooper'
        'spencer'

        This method sets the `results` attribute to be a pandas data frame
        containing the results of the study.

        :param delta_method: 'cooper', 'spencer'.
        :return: None.
        """

        delta_method = np.atleast_1d(delta_method)
        # calculate hour angle
        self.omega = calculate_omega(
            self.day_of_year, self.data_sampling, self.lon_input, self.gmt_offset
        )
        # fit daily signal of cos theta
        self.scale_factor_costheta, self.costheta_fit = find_fit_costheta(
            self.data_matrix, self.clear_index
        )
        # estimate declination angles
        self.delta_cooper = delta_cooper(self.day_of_year, self.daily_meas)
        self.delta_spencer = delta_spencer(self.day_of_year, self.daily_meas)
        # initialize parameters
        if self.init_values is not None:
            lat_initial = self.init_values[0]
            tilt_initial = self.init_values[1]
            azim_initial = self.init_values[2]
        else:
            if self.nrandom is None:
                lat_initial = [10]
                tilt_initial = [10]
                azim_initial = [10]
            else:
                lat_initial, tilt_initial, azim_initial = random_initial_values(
                    self.nrandom
                )

        counter = 0
        self.create_results_table()
        for x1 in self.threshold_x1:
            # first quantile value in signal decomposition algorithm
            for x2 in self.threshold_x2:
                # second quantile value in signal decomposition algorithm
                filtered_data = filter_data(
                    self.data_matrix, self.daytime_threshold, x1, x2
                )
                for delta_id in delta_method:
                    # declination angle
                    if delta_id in ("Cooper", "cooper"):
                        delta = self.delta_cooper
                    if delta_id in ("Spencer", "spencer"):
                        delta = self.delta_spencer
                    for day_range_id in self.day_range_dict:
                        # day range
                        day_interval = self.day_range_dict[day_range_id]
                        boolean_filter = self.get_day_range(filtered_data, day_interval)
                        delta_f = delta[boolean_filter]
                        omega_f = self.omega[boolean_filter]
                        if ~np.any(boolean_filter):
                            print("No data made it through filters")
                        # choose function and unknowns based on provided inputs
                        # choose range for each unknown
                        func_customized, bounds = select_function(
                            self.lat_input, self.tilt_input, self.azimuth_input
                        )
                        # create dictionary keys for unknowns. If an input value for lat, tilt or azimuth is not
                        # provided, it will be tagged as an unknown
                        dict_keys = determine_keys(
                            latitude=self.lat_input,
                            tilt=self.tilt_input,
                            azimuth=self.azimuth_input,
                        )
                        nvalues = len(lat_initial)

                        for init_val_ix in np.arange(nvalues):
                            # loop over initial values
                            init_values_dict = {
                                "latitude": lat_initial[init_val_ix],
                                "tilt": tilt_initial[init_val_ix],
                                "azimuth": azim_initial[init_val_ix],
                            }
                            init_values, ivr = select_init_values(
                                init_values_dict, dict_keys
                            )
                            try:
                                # estimate latitude and/or tilt and/or azimuth. If parameter is in keys, it will be
                                # estimated
                                estimates = run_curve_fit(
                                    func=func_customized,
                                    keys=dict_keys,
                                    delta=delta_f,
                                    omega=omega_f,
                                    costheta=self.costheta_fit,
                                    boolean_filter=boolean_filter,
                                    init_values=init_values,
                                    fit_bounds=bounds,
                                )
                            except RuntimeError:
                                input_array = np.array(
                                    [
                                        self.lat_input,
                                        self.tilt_input,
                                        self.azimuth_input,
                                    ]
                                )
                                estimates = np.full(np.sum(input_array == None), np.nan)
                            # create dictionary with dict_keys and estimates
                            estimates_dict = dict(zip(dict_keys, estimates))
                            # dynamic results dataFrame based on provided inputs
                            lat = (
                                estimates_dict["latitude_estimate"]
                                if "latitude_estimate" in estimates_dict
                                else self.lat_input
                            )
                            tilt = (
                                estimates_dict["tilt_estimate"]
                                if "tilt_estimate" in estimates_dict
                                else self.tilt_input
                            )
                            azim = (
                                estimates_dict["azimuth_estimate"]
                                if "azimuth_estimate" in estimates_dict
                                else self.azimuth_input
                            )

                            self.costheta_estimated = calculate_costheta(
                                func=func_costheta,
                                delta=delta,
                                omega=self.omega,
                                lat=lat,
                                tilt=tilt,
                                azim=azim,
                            )
                            # calculate cos theta from analytical equation in case ground truth values are provided
                            if None not in (
                                self.lat_true_value,
                                self.tilt_true_value,
                                self.azimuth_true_value,
                            ):
                                self.costheta_ground_truth = calculate_costheta(
                                    func=func_costheta,
                                    delta=delta,
                                    omega=self.omega,
                                    lat=self.lat_true_value,
                                    tilt=self.tilt_true_value,
                                    azim=self.azimuth_true_value,
                                )

                            self.results.loc[counter] = (
                                [day_range_id, delta_id, x1, x2] + ivr + list(estimates)
                            )
                            counter += 1

        if self.lat_true_value is not None and self.lat_input is None:
            self.results["latitude residual"] = (
                self.lat_true_value - self.results["latitude"]
            )
        if self.tilt_true_value is not None and self.tilt_input is None:
            self.results["tilt residual"] = self.tilt_true_value - self.results["tilt"]
        if self.azimuth_true_value is not None and self.azimuth_input is None:
            self.results["azimuth residual"] = (
                self.azimuth_true_value - self.results["azimuth"]
            )
        return

    def get_day_range(self, input_data, interval):
        """
        This method was intended to evaluate different day ranges for the estimation of tilt and  azimuth. However, no
        gain was seen from using day ranges instead of the full year. Therefore, the study is usually run with
        `interval=None'
        :param input_data: boolean array containing daytime hours
        :param interval:  day interval to be used in estimation
        :return: Boolean DataFrame with day selection
        """
        if interval is not None:
            day_range = (self.day_of_year > interval[0]) & (
                self.day_of_year < interval[1]
            )
        else:
            day_range = np.ones(self.day_of_year.shape, dtype=bool)
        output = input_data * self.clear_index * day_range
        return output

    def create_results_table(self):
        cols = [
            "day range",
            "declination method",
            "cvx parameter",
            "threshold quantile",
            "latitude initial value",
            "tilt initial value",
            "azimuth initial value",
        ]
        if self.lat_input is None:
            cols.append("latitude")
        if self.tilt_input is None:
            cols.append("tilt")
        if self.azimuth_input is None:
            cols.append("azimuth")
        self.results = pd.DataFrame(columns=cols)
