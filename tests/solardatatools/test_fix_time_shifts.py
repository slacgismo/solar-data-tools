import unittest
import sys
from pathlib import Path
import numpy as np
import cvxpy as cvx
from solardatatools.algorithms import TimeShift


class TestFixTimeShift(unittest.TestCase):

    def setUp(self):
        np.set_printoptions(threshold=sys.maxsize)
        self.maxDiff = None

    def test_fix_time_shifts(self):
        filepath = Path(__file__).parent.parent
        input_power_signals_file_path = \
            filepath / 'fixtures' / 'time_shifts' / \
            'two_year_signal_with_shift.csv'
        with open(input_power_signals_file_path) as file:
            power_data_matrix = np.loadtxt(file, delimiter=',')

        use_days_file_path = \
            filepath / 'fixtures' / 'time_shifts' / 'clear_days.csv'
        with open(use_days_file_path) as file:
            use_days = np.loadtxt(file, delimiter=',')

        output_power_signals_file_path = \
            filepath / 'fixtures' / 'time_shifts' / 'two_year_signal_fixed.csv'
        with open(output_power_signals_file_path) as file:
            expected_power_data_fix = np.loadtxt(file, delimiter=',')

        # Underling solar-data-tools uses MOSEK solver and
        # if it's not used, try with ECOS solver.
        # However, fails with ECOS solver and raises cvx.SolverError.
        try:
            time_shift_analysis = TimeShift()
            time_shift_analysis.run(
                power_data_matrix, use_ixs=use_days, solver='MOSEK'
            )
            actual_power_data_fix = time_shift_analysis.corrected_data
        except (cvx.SolverError, ValueError):
            self.skipTest("This test uses MOSEK solver"
                + "because default ECOS solver fails with large data. "
                + "Unless MOSEK is installed, this test fails.")
        else:
            np.testing.assert_almost_equal(
                actual_power_data_fix,
                expected_power_data_fix,
                decimal=.2
            )
