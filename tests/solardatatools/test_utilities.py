import unittest
import os
import numpy as np
from solardatatools.utilities import local_median_regression_with_seasonal


class TestCVXFilters(unittest.TestCase):

    def test_local_median_regression_with_seasonal(self):
        data_file_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__),
            "../fixtures/utilities/corrupt_seasonal_signal.csv"))
        with open(data_file_path) as file:
            data = np.loadtxt(file, delimiter=',')
        expected_data_file_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__),
            "../fixtures/utilities/local_median_seasonal_filter.csv"))
        with open(expected_data_file_path) as file:
            expected_output = np.loadtxt(file, delimiter=',')
        actual_output = local_median_regression_with_seasonal(data)
        np.testing.assert_array_equal(expected_output,
                                      actual_output)