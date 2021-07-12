import unittest
from pathlib import Path
import numpy as np
from solardatatools.signal_decompositions import l1_l2d2p365
from solardatatools.utilities import basic_outlier_filter


class TestCVXFilters(unittest.TestCase):

    def test_local_median_regression_with_seasonal(self):
        filepath = Path(__file__).parent.parent
        data_file_path = \
            filepath / 'fixtures' / 'utilities' / 'corrupt_seasonal_signal.csv'
        with open(data_file_path) as file:
            data = np.loadtxt(file, delimiter=',')
        expected_data_file_path = \
            filepath / 'fixtures' / 'utilities' / \
            'local_median_seasonal_filter.csv'
        with open(expected_data_file_path) as file:
            expected_output = np.loadtxt(file, delimiter=',')
        actual_output = l1_l2d2p365(data)
        np.testing.assert_array_almost_equal(
            expected_output, actual_output, decimal=1
        )

class TestOutlierFilter(unittest.TestCase):

    def test_basic_outlier_filter(self):
        np.random.seed(42)
        x = np.random.normal(size=20)
        x[0] *= 5
        msk = basic_outlier_filter(x)
        self.assertEqual(np.sum(~msk), 1)
        self.assertAlmostEqual(x[~msk][0], 2.4835707650561636)

if __name__ == '__main__':
    unittest.main()