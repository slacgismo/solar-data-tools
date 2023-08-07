import unittest
from pathlib import Path
import numpy as np
import cvxpy as cvx
from solardatatools.clear_day_detection import ClearDayDetection


class TestClearDayDetection(unittest.TestCase):
    def test_find_clear_days(self):
        filepath = Path(__file__).parent.parent
        data_file_path = (
            filepath
            / "fixtures"
            / "clear_day_detection"
            / "one_year_power_signals_1.csv"
        )
        with open(data_file_path) as file:
            data = np.loadtxt(file, delimiter=",")
        expected_data_file_path = (
            filepath / "fixtures" / "clear_day_detection" / "one_year_weights_1.csv"
        )
        with open(expected_data_file_path) as file:
            expected_output = np.loadtxt(file, delimiter=",")
        expected_output = expected_output >= 1e-3

        clear_day_detection = ClearDayDetection()
        actual_output = clear_day_detection.find_clear_days(data)

        np.testing.assert_array_equal(expected_output, actual_output)

    def test_clear_day_weights(self):
        filepath = Path(__file__).parent.parent
        data_file_path = (
            filepath
            / "fixtures"
            / "clear_day_detection"
            / "one_year_power_signals_1.csv"
        )
        with open(data_file_path) as file:
            data = np.loadtxt(file, delimiter=",")
        expected_data_file_path = (
            filepath / "fixtures" / "clear_day_detection" / "one_year_weights_1.csv"
        )
        with open(expected_data_file_path) as file:
            expected_output = np.loadtxt(file, delimiter=",")

        clear_day_detection = ClearDayDetection()
        actual_output = clear_day_detection.find_clear_days(data, boolean_out=False)

        np.testing.assert_array_almost_equal(expected_output, actual_output, 4)
