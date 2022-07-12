import unittest
import os
from pathlib import Path
import numpy as np
path = Path.cwd().parent.parent
os.chdir(path)
from pvsystemprofiler.algorithms.longitude.calculation import calculate_longitude


class TestCalculateLongitude(unittest.TestCase):

    def test_calculate_longitude(self):
        # INPUTS
        filepath = Path(__file__).parent.parent
        # eot_duffie
        eot_duffie_file_path = filepath / "fixtures" / "longitude" / "eot_duffie_output.csv"
        with open(eot_duffie_file_path) as file:
                eot_duffie = np.genfromtxt(file, delimiter=',')
        # solarnoon
        solarnoon_file_path = filepath / "fixtures" / "longitude" / "solarnoon.csv"
        with open(solarnoon_file_path) as file:
                solarnoon = np.genfromtxt(file, delimiter=',')
        # days
        days_file_path = filepath / "fixtures" / "longitude" / "days.csv"
        with open(days_file_path) as file:
                days = np.genfromtxt(file, delimiter=',')
                days = days.astype(dtype=bool)
        # gmt_offset
        gmt_offset = -5
        # loss
        loss = 'l2'

        # Expected Longitude Output is generated in tests/fixtures/longitude/longitude_fitting_and_calculation_test_data_creator.ipynb
        expected_output =  -77.10636729272031

        actual_output = calculate_longitude(eot_duffie, solarnoon, days, gmt_offset)
        np.testing.assert_almost_equal(actual_output, expected_output, decimal=1)


if __name__ == '__main__':
    unittest.main()
