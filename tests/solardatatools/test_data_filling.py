import unittest
from pathlib import Path
import numpy as np
from solardatatools.data_filling import zero_nighttime, interp_missing


class TestDataFilling(unittest.TestCase):
    def test_zero_nighttime(self):
        data_file_path = Path(__file__).parent.parent.joinpath(
            "fixtures/data_filling/pvdaq_2d_data_input.csv"
        )
        expected_data_file_path = Path(__file__).parent.parent.joinpath(
            "fixtures/data_filling/expected_zero_nighttime_output.csv"
        )

        with open(data_file_path) as file:
            input_data = np.genfromtxt(file, delimiter=",")
        with open(expected_data_file_path) as file:
            expected_output = np.genfromtxt(file, delimiter=",")

        actual_output = zero_nighttime(input_data)
        np.testing.assert_array_almost_equal(actual_output, expected_output)

    def test_interp_missing(self):
        # using zero_nighttime expected output as interp_missing() input
        data_file_path = data_file_path = Path(__file__).parent.parent.joinpath(
            "fixtures/data_filling/expected_zero_nighttime_output.csv"
        )
        expected_data_file_path = Path(__file__).parent.parent.joinpath(
            "fixtures/data_filling/expected_interp_missing_output.csv"
        )

        with open(data_file_path) as file:
            input_data = np.genfromtxt(file, delimiter=",")
        with open(expected_data_file_path) as file:
            expected_output = np.genfromtxt(file, delimiter=",")

        actual_output = interp_missing(input_data)
        np.testing.assert_array_almost_equal(actual_output, expected_output)


if __name__ == "__main__":
    unittest.main()
