import unittest
import os
from pathlib import Path
import numpy as np
path = Path.cwd().parent.parent
os.chdir(path)
from pvsystemprofiler.utilities.equation_of_time import eot_da_rosa, eot_duffie


class TestEquationOfTime(unittest.TestCase):

    # importing input for both eot tests
    filepath = Path(__file__).parent.parent
    input_data_file_path = filepath / 'fixtures' / 'longitude' / 'eot_input.csv'
    with open(input_data_file_path) as file:
            input_data = np.genfromtxt(file, delimiter=',')


    def test_eot_duffie(self):

        expected_data_file_path = self.filepath / 'fixtures' / 'longitude' / 'eot_duffie_output.csv'
        with open(expected_data_file_path) as file:
            expected_output = np.genfromtxt(file, delimiter=',')

        actual_output = eot_duffie(self.input_data)
        np.testing.assert_array_almost_equal(actual_output, expected_output)


    def test_eot_da_rosa(self):

        expected_data_file_path = self.filepath / 'fixtures' / 'longitude' / 'eot_da_rosa_output.csv'
        with open(expected_data_file_path) as file:
            expected_output = np.genfromtxt(file, delimiter=',')

        actual_output = eot_da_rosa(self.input_data)
        np.testing.assert_array_almost_equal(actual_output, expected_output)


if __name__ == '__main__':
    unittest.main()
