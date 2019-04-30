import unittest
import sys
import os
import numpy as np
import cvxpy as cvx
from solardatatools.data_transforms import fix_time_shifts

class TestFixTimeShift(unittest.TestCase):

    def setUp(self):
        np.set_printoptions(threshold=sys.maxsize)
        self.maxDiff = None

    def test_fix_time_shifts(self):

        input_power_signals_file_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__),
                "../fixtures/time_shifts",
                "one_year_power_signals_1.csv"))
        with open(input_power_signals_file_path) as file:
            power_data_matrix = np.loadtxt(file, delimiter=',')

        output_power_signals_file_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__),
                "../fixtures/time_shifts",
                "one_year_power_signals_time_shift_fix_1.csv"))
        with open(output_power_signals_file_path) as file:
            expected_power_data_fix = np.loadtxt(file, delimiter=',')

        # Underling solar-data-tools uses MOSEK solver and
        # if it's not used, try with ECOS solver.
        # However, fails with ECOS solver and raises cvx.SolverError.
        try:
            actual_power_data_fix = fix_time_shifts(power_data_matrix)
        except (cvx.SolverError, ValueError):
            self.skipTest("This test uses MOSEK solver"
                + "because default ECOS solver fails with large data. "
                + "Unless MOSEK is installed, this test fails.")
        else:
            np.testing.assert_array_equal(actual_power_data_fix,
                                          expected_power_data_fix)
