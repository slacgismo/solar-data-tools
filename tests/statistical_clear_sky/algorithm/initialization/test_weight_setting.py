import unittest
import os
import numpy as np
import cvxpy as cvx
from statistical_clear_sky.algorithm.initialization.weight_setting\
 import WeightSetting

class TestWeightSetting(unittest.TestCase):

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

        weight_setting = WeightSetting()
        actual_weights = weight_setting.obtain_weights(power_signals_d)
        np.testing.assert_allclose(actual_weights, expected_weights, rtol=1e-5)
