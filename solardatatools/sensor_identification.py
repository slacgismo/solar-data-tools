""" Sensor Identification Module

This module contains a class for choosing which irradiance sensor best
describes a PV power or current data set. We assume a linear model between
irradiance and power/current, and we use k-fold cross validation to assess
which irradiance sensor provides the best predictive power.

Generally speaking, we can try to assess a sensor's distance from an array and
its plane-of-array mismatch. Hopefully, there exists a sensor that is both
close by the array and well aligned; however, this is not always the case.
We use clear sky data to assess POA mismatch and cloudy sky data to assess
distance from array. If there is a discrepancy in which sensor is "best" under
these two data filtering schemes, the algorithm alerts the user.

"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.model_selection import KFold, TimeSeriesSplit

def rmse(residuals):
    return np.sqrt(np.average(np.power(residuals, 2)))

class SensorIdentification():
    def __init__(self, data_handler_obj):
        """
        This class is always instantiated with a DataHandler class instance.
        This instance should have the pipeline run, with extra columns included
        that identify the candidate sensors.

        :param data_handler_obj: (required) a DataHandler class instance
        """
        self.data_handler = data_handler_obj
        self.sensor_keys = np.array(list(self.data_handler.extra_matrices.keys()))
        if len(self.sensor_keys) == 0:
            print('Please add sensor columns to DataHandler pipeline.')
            return
        self.coverage_scores = self.data_handler.extra_quality_scores
        nan_masks = [~np.isnan(m[1])
                     for m in self.data_handler.extra_matrices.items()]
        self.compare_mask = np.alltrue(np.array(nan_masks), axis=0)
        # These attributes are set when running the identify method
        self.results_table = None
        self.chosen_sensor = None
        self.consistent_answer = None

    def identify(self, n_splits=20, model='least-squares', epsilon=1.35,
                 max_iter=100, alpha=0.0001):
        # least squares is about 10x faster than huber, but less robust
        if model == 'least-squares':
            lr = LinearRegression()
        elif model == 'huber':
            lr = HuberRegressor(epsilon=epsilon, max_iter=max_iter,
                                alpha=alpha)
        self.results_table = None
        self.consistent_answer = None
        self.chosen_sensor = None
        filters = {
            'no_errors': self.data_handler.daily_flags.no_errors,
            'clear': self.data_handler.daily_flags.clear,
            'cloudy': self.data_handler.daily_flags.cloudy
        }
        results = pd.DataFrame(
            columns=['sensor', 'filter', 'corr', 'cv-rmse', 'cv-mbe']
        )
        counter = 0
        for filter_key, filter in filters.items():
            mask = np.zeros_like(self.compare_mask, dtype=bool)
            mask[:, filter] = self.compare_mask[:, filter]
            # Form (x, y) data
            y = self.data_handler.filled_data_matrix[mask]
            Xs = {key: matrix[mask].reshape(-1, 1)
                  for key, matrix in self.data_handler.extra_matrices.items()}

            for key, data in Xs.items():
                # Calculate correlation coefficient
                corr = np.corrcoef(data.squeeze(), y)[0, 1]
                # Get k-fold splits for cross validation
                splits = TimeSeriesSplit(
                    n_splits=n_splits
                ).split(data)
                residuals = []
                # CV of linear model to predict system output from measured irradiance
                for train_ix, test_ix in splits:
                    try:
                        # Fit linear regression model for each data split
                        fit = lr.fit(data[train_ix], y[train_ix])
                        # Get predictions for holdout set
                        y_pred = fit.predict(data[test_ix])
                        # Calculate the residuals
                        residuals.append(y[test_ix] - y_pred)
                    except:
                        residuals.append(np.inf)
                # Collect the residuals from all the splits
                residuals = np.concatenate(residuals)
                # Calculate statistics and store results
                results.loc[counter] = (
                    key, filter_key, corr, rmse(residuals), np.mean(residuals)
                )
                counter += 1
        self.results_table = results
        gb = results.groupby('filter')
        lowest_error = gb['cv-rmse'].agg(np.argmin)
        # All three tests (no error, clear, and cloudy) give same result
        if len(set(lowest_error)) == 1:
            self.chosen_sensor = self.sensor_keys[int(lowest_error.iloc[0])]
            self.consistent_answer = True
        else:
            ixs = lowest_error.astype(int).values
            sensors = self.sensor_keys[ixs]
            self.chosen_sensor = dict(zip(lowest_error.index, sensors))
            self.consistent_answer = False
        return
