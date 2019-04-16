import unittest
import os
import numpy as np
from solardatatools.clear_day_detection import find_clear_days

class TestClearDayDetection(unittest.TestCase):

    def test_find_clear_days(self):
        directory = os.getcwd()
        print(directory)
        data = np.loadtxt('../fixtures/one_year_power_signals_1.csv',
                          delimiter=',')
        expected_output = np.loadtxt('../fixtures/one_year_weights_1.csv',
                                     delimiter=',')
        expected_output = expected_output >= 1e-3
        actual_output = find_clear_days(data)
        np.testing.assert_array_equal(expected_output,
                                      actual_output)
