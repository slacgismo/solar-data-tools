import unittest
import os
import numpy as np
import cvxpy as cvx
from statistical_clear_sky.algorithm.initialization.weight_setting\
 import WeightSetting

class TestWeightSetting(unittest.TestCase):

    def test_obtain_weights(self):

        power_signals_d = np.array([[3.65099996e-01, 0.00000000e+00,
                                     0.00000000e+00, 2.59570003e+00],
                                    [6.21100008e-01, 0.00000000e+00,
                                     0.00000000e+00, 2.67740011e+00],
                                    [8.12500000e-01, 0.00000000e+00,
                                     0.00000000e+00, 2.72729993e+00],
                                    [9.00399983e-01, 0.00000000e+00,
                                     0.00000000e+00, 2.77419996e+00]])

        expected_weights = np.array([0.0, 0.0, 0.0, 0.0])

        weight_setting = WeightSetting()
        actual_weights = weight_setting.obtain_weights(power_signals_d)

        np.testing.assert_array_equal(actual_weights, expected_weights)

    def test_obtain_weights_with_large_data(self):

        input_power_signals_file_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__),
            "../../fixtures/initialization/one_year_power_signals_1.csv"))
        with open(input_power_signals_file_path) as file:
            power_signals_d = np.loadtxt(file, delimiter=',')

        weights_file_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__),
                "../../fixtures/initialization/one_year_weights_1.csv"))
        with open(weights_file_path) as file:
            expected_weights = np.loadtxt(file, delimiter=',')

        weight_setting = WeightSetting(solver_type='MOSEK')
        try:
            actual_weights = weight_setting.obtain_weights(power_signals_d)
        except cvx.SolverError:
            self.skipTest("This test uses MOSEK solver"
                + "because default ECOS solver fails with large data. "
                + "Unless MOSEK is installed, this test fails.")
        else:
            np.testing.assert_allclose(actual_weights, expected_weights, rtol=1e-5)
