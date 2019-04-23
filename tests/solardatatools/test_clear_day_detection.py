import unittest
import os
import numpy as np
from solardatatools.clear_day_detection import find_clear_days

class TestClearDayDetection(unittest.TestCase):

    def test_find_clear_days(self):

        data_file_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__),
            "../fixtures/clear_day_detection/one_year_power_signals_1.csv"))
        with open(data_file_path) as file:
            data = np.loadtxt(file, delimiter=',')
        expected_data_file_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__),
            "../fixtures/clear_day_detection/one_year_weights_1.csv"))
        with open(expected_data_file_path) as file:
            expected_output = np.loadtxt(file, delimiter=',')
        expected_output = expected_output >= 1e-3
        actual_output = find_clear_days(data)
        np.testing.assert_array_equal(expected_output,
                                      actual_output)

    def test_clear_day_weights(self):
        data_file_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__),
            "../fixtures/clear_day_detection/one_year_power_signals_1.csv"))
        with open(data_file_path) as file:
            data = np.loadtxt(file, delimiter=',')
        expected_data_file_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__),
            "../fixtures/clear_day_detection/one_year_weights_1.csv"))
        with open(expected_data_file_path) as file:
            expected_output = np.loadtxt(file, delimiter=',')
        actual_output = find_clear_days(data, boolean_out=False)
        np.testing.assert_array_equal(expected_output,
                                      actual_output)
