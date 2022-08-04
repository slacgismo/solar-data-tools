import unittest
import os
from pathlib import Path
import numpy as np
path = Path.cwd().parent.parent
os.chdir(path)
from pvsystemprofiler.algorithms.angle_of_incidence.lambda_functions import select_function
from pvsystemprofiler.algorithms.angle_of_incidence.curve_fitting import run_curve_fit

class TestFitAzimuth(unittest.TestCase):

    def test_fit_azimuth(self):
        # INPUTS

        filepath = Path(__file__).parent.parent
        # delta_f
        filepath = Path(__file__).parent.parent
        delta_f_file_path = filepath / "fixtures" / "tilt_azimuth" / "delta_f.csv"
        with open(delta_f_file_path) as file:
                delta_f = np.genfromtxt(file, delimiter=',')
        # omega_f
        filepath = Path(__file__).parent.parent
        omega_f_file_path = filepath / "fixtures" / "tilt_azimuth" / "omega_f.csv"
        with open(omega_f_file_path) as file:
                omega_f = np.genfromtxt(file, delimiter=',')
        # costheta_fit
        filepath = Path(__file__).parent.parent
        costheta_fit_file_path = filepath / "fixtures" / "tilt_azimuth" / "costheta_fit.csv"
        with open(costheta_fit_file_path) as file:
                costheta_fit = np.genfromtxt(file, delimiter=',')
        # boolean_filter
        filepath = Path(__file__).parent.parent
        boolean_filter_file_path = filepath / "fixtures" / "tilt_azimuth" / "boolean_filter.csv"
        with open(boolean_filter_file_path) as file:
                boolean_filter = np.genfromtxt(file, delimiter=',')
                boolean_filter = boolean_filter.astype(dtype=bool)
        # keys
        keys = ['tilt', 'azimuth']
        # init_values
        init_values = [30, 30]

        # Expected Tilt and azimuth output is generated in tests/fixtures/tilt_azimuth/tilt_azimuth_Estimation_data_creator.ipynb
        expected_output = 1.654457422429566
        func_customized, bounds = select_function(39.4856, None, None)     
        actual_output = run_curve_fit(func=func_customized, keys=['tilt', 'azimuth'], delta=delta_f, omega=omega_f, costheta=costheta_fit, boolean_filter=boolean_filter, init_values=[30, 30], fit_bounds=bounds)[1]
        np.testing.assert_almost_equal(actual_output, expected_output, decimal=4)

if __name__ == '__main__':
    unittest.main()
