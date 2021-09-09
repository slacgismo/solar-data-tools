import unittest
from pathlib import Path
import numpy as np
from solardatatools.algorithms.soiling import soiling_seperation


class TestSoilingSeperation(unittest.TestCase):
    def test_soiling_seperation(self):
        # inputs for soiling seperation
        bix_path = Path(__file__).parent.parent.joinpath(
            "fixtures/soiling_seperation/bix_soiling_seperation_input.csv"
        )
        pi_path = Path(__file__).parent.parent.joinpath(
            "fixtures/soiling_seperation/pi_soiling_seperation_input.csv"
        )

        # expected outputs
        expected_s1_path = Path(__file__).parent.parent.joinpath(
            "fixtures/soiling_seperation/s1_soiling_seperation_output.csv"
        )
        expected_s2_path = Path(__file__).parent.parent.joinpath(
            "fixtures/soiling_seperation/s2_soiling_seperation_output.csv"
        )
        expected_s3_path = Path(__file__).parent.parent.joinpath(
            "fixtures/soiling_seperation/s3_soiling_seperation_output.csv"
        )

        with open(bix_path) as file:
            bix = np.genfromtxt(file, delimiter=",")
        with open(pi_path) as file:
            pi = np.genfromtxt(file, delimiter=",")
        with open(expected_s1_path) as file:
            expected_s1 = np.genfromtxt(file, delimiter=",")
        with open(expected_s2_path) as file:
            expected_s2 = np.genfromtxt(file, delimiter=",")
        with open(expected_s3_path) as file:
            expected_s3 = np.genfromtxt(file, delimiter=",")

        expected_output = [expected_s1, expected_s2, expected_s3]
        actual_s1, actual_s2, actual_s3 = soiling_seperation(
            pi, index_set=bix, iterations=1
        )
        actual_output = [actual_s1, actual_s2, actual_s3]
        np.testing.assert_array_almost_equal(actual_s1, expected_s1)
        np.testing.assert_array_almost_equal(actual_s2, expected_s2)
        np.testing.assert_array_almost_equal(actual_s3, expected_s3)


if __name__ == "__main__":
    unittest.main()
