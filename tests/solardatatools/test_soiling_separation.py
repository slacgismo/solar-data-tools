import unittest
from pathlib import Path
import numpy as np
from solardatatools.algorithms.soiling import soiling_separation


class TestSoilingSeparation(unittest.TestCase):
    def test_soiling_separation(self):
        # inputs for soiling separation
        bix_path = Path(__file__).parent.parent.joinpath(
            "fixtures/soiling_separation/bix_soiling_separation_input.csv"
        )
        pi_path = Path(__file__).parent.parent.joinpath(
            "fixtures/soiling_separation/pi_soiling_separation_input.csv"
        )

        # expected outputs
        expected_s1_path = Path(__file__).parent.parent.joinpath(
            "fixtures/soiling_separation/s1_soiling_separation_output.csv"
        )
        expected_s2_path = Path(__file__).parent.parent.joinpath(
            "fixtures/soiling_separation/s2_soiling_separation_output.csv"
        )
        expected_s3_path = Path(__file__).parent.parent.joinpath(
            "fixtures/soiling_separation/s3_soiling_separation_output.csv"
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

        out = soiling_separation(pi, index_set=bix, iterations=1)

        expected_output = [expected_s1, expected_s2, expected_s3]
        actual_output = [out["soiling"], out["seasonal"], out["trend"]]
        np.testing.assert_array_almost_equal(actual_output, expected_output)
