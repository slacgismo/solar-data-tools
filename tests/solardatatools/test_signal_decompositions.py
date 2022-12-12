import unittest
from pathlib import Path
import pandas as pd
import numpy as np
from solardatatools import signal_decompositions as sd


class TestSignalDecompositions(unittest.TestCase):
    # Tolerance for difference between solutions
    tolerance = 7  # is this reasonable?

    def assertListAlmostEqual(self, list1, list2, tol=tolerance):
        self.assertEqual(len(list1), len(list2))
        for a, b in zip(list1, list2):
            self.assertAlmostEqual(a, b, tol)

    def test_l2_l1d1_l2d2p365_default(self):
        """Test with default args"""

        fname = "test_l2_l1d1_l2d2p365_data_input.csv"
        cvxpy_solver = "MOSEK" # scs is only other option for tests as of 11/28/22

        # Test signal data; incl MOSEK and SCS expected solutions
        filepath = Path(__file__).parent.parent
        data_file_path = (
            filepath / "fixtures" / "signal_decompositions" / fname
        )
        test_data = pd.read_csv(data_file_path)

        # Raw signal
        signal = test_data["test_signal"].array[:365]
        # Expected output
        expected_s_hat = test_data[f"expected_s_hat_{cvxpy_solver.lower()}_365"].array.dropna()
        expected_s_seas = test_data[f"expected_s_seas_{cvxpy_solver.lower()}_365"].array.dropna()

        # Run test with default args
        actual_s_hat, actual_s_seas = sd.l2_l1d1_l2d2p365(signal, solver=cvxpy_solver)

        self.assertListAlmostEqual(list(expected_s_hat), list(actual_s_hat))
        self.assertListAlmostEqual(list(expected_s_seas), list(actual_s_seas))


    def test_l2_l1d1_l2d2p365_tv_weights(self):
        """Test with TV weights"""

        fname = "test_l2_l1d1_l2d2p365_data_input.csv"
        cvxpy_solver = "MOSEK" # scs is only other option for tests as of 11/28/22

        # Test signal data; incl MOSEK and SCS expected solutions
        filepath = Path(__file__).parent.parent
        data_file_path = (
            filepath / "fixtures" / "signal_decompositions" / fname
        )
        test_data = pd.read_csv(data_file_path)

        # Raw signal
        signal = test_data["test_signal"].array[:365]
        # Expected output
        expected_s_hat = test_data[f"expected_s_hat_{cvxpy_solver.lower()}_tvw_365"].array.dropna()
        expected_s_seas = test_data[f"expected_s_seas_{cvxpy_solver.lower()}_tvw_365"].array.dropna()

        # Run test
        rand_tv_weights = test_data["rand_tv_weights_365"].dropna() # len(signal)-1
        actual_s_hat, actual_s_seas = sd.l2_l1d1_l2d2p365(
            signal,
            solver=cvxpy_solver,
            tv_weights=rand_tv_weights
        )

        self.assertListAlmostEqual(list(expected_s_hat), list(actual_s_hat))
        self.assertListAlmostEqual(list(expected_s_seas), list(actual_s_seas))

    def test_l2_l1d1_l2d2p365_residual_weights(self):
        """Test with residual weights"""

        fname = "test_l2_l1d1_l2d2p365_data_input.csv"
        cvxpy_solver = "MOSEK"  # scs is only other option for tests as of 11/28/22

        # Test signal data; incl MOSEK and SCS expected solutions
        filepath = Path(__file__).parent.parent
        data_file_path = (
                filepath / "fixtures" / "signal_decompositions" / fname
        )
        test_data = pd.read_csv(data_file_path)

        # Raw signal
        signal = test_data["test_signal"].array[:365]
        # Expected output
        expected_s_hat = test_data[f"expected_s_hat_{cvxpy_solver.lower()}_rw_365"].array.dropna()
        expected_s_seas = test_data[f"expected_s_seas_{cvxpy_solver.lower()}_rw_365"].array.dropna()

        # Run test
        rand_residual_weights = test_data["rand_residual_weights_365"].dropna()
        actual_s_hat, actual_s_seas = sd.l2_l1d1_l2d2p365(
            signal,
            solver=cvxpy_solver,
            residual_weights=rand_residual_weights
        )

        self.assertListAlmostEqual(list(expected_s_hat), list(actual_s_hat))
        self.assertListAlmostEqual(list(expected_s_seas), list(actual_s_seas))

    def test_l2_l1d1_l2d2p365_transition(self):
        """Test with non-default transition location"""

        transition  = 100
        fname = "test_l2_l1d1_l2d2p365_data_input.csv"
        cvxpy_solver = "MOSEK"  # scs is only other option for tests as of 11/28/22

        # Test signal data; incl MOSEK and SCS expected solutions
        filepath = Path(__file__).parent.parent
        data_file_path = (
                filepath / "fixtures" / "signal_decompositions" / fname
        )
        test_data = pd.read_csv(data_file_path)

        # Raw signal
        signal = test_data["test_signal"].array[:365]
        # Expected output
        expected_s_hat = test_data[f"expected_s_hat_{cvxpy_solver.lower()}_transition_100_365"].array.dropna()
        expected_s_seas = test_data[f"expected_s_seas_{cvxpy_solver.lower()}_transition_100_365"].array.dropna()

        # Run test
        rand_weights = np.random.uniform(10, 200, len(signal)-1)
        actual_s_hat, actual_s_seas = sd.l2_l1d1_l2d2p365(
            signal,
            transition_locs=transition
        )

        self.assertListAlmostEqual(list(expected_s_hat), list(actual_s_hat))
        self.assertListAlmostEqual(list(expected_s_seas), list(actual_s_seas))

    def test_l2_l1d1_l2d2p365_default_long(self):
        """Test with default args and signal with len >365"""

        fname = "test_l2_l1d1_l2d2p365_data_input.csv"
        cvxpy_solver = "MOSEK"  # scs is only other option for tests as of 11/28/22

        # Test signal data; incl MOSEK and SCS expected solutions
        filepath = Path(__file__).parent.parent
        data_file_path = (
                filepath / "fixtures" / "signal_decompositions" / fname
        )
        test_data = pd.read_csv(data_file_path)

        # Raw signal
        signal = test_data["test_signal"].array
        # Expected output
        expected_s_hat = test_data[f"expected_s_hat_{cvxpy_solver.lower()}"].array
        expected_s_seas = test_data[f"expected_s_seas_{cvxpy_solver.lower()}"].array

        # Run test with default args
        actual_s_hat, actual_s_seas = sd.l2_l1d1_l2d2p365(signal, solver=cvxpy_solver)

        self.assertListAlmostEqual(list(expected_s_hat), list(actual_s_hat))
        self.assertListAlmostEqual(list(expected_s_seas), list(actual_s_seas))

    def test_l2_l1d1_l2d2p365_idx_select(self):
        """Test with signal with select indices"""

        fname = "test_l2_l1d1_l2d2p365_data_input.csv"
        cvxpy_solver = "MOSEK"  # scs is only other option for tests as of 11/28/22

        # Test signal data; incl MOSEK and SCS expected solutions
        filepath = Path(__file__).parent.parent
        data_file_path = (
                filepath / "fixtures" / "signal_decompositions" / fname
        )
        test_data = pd.read_csv(data_file_path)

        # Take first 300 days of dataset
        indices = list([True] * 300) + list([False] * (730 - 300))

        # Raw signal
        signal = test_data["test_signal"].array
        # Expected output
        expected_s_hat = test_data[f"expected_s_hat_{cvxpy_solver.lower()}_ixs"].array
        expected_s_seas = test_data[f"expected_s_seas_{cvxpy_solver.lower()}_ixs"].array

        # Run test with default args
        actual_s_hat, actual_s_seas = sd.l2_l1d1_l2d2p365(
            signal,
            solver=cvxpy_solver,
            use_ixs=indices
        )

        self.assertListAlmostEqual(list(expected_s_hat), list(actual_s_hat))
        self.assertListAlmostEqual(list(expected_s_seas), list(actual_s_seas))

    def test_l2_l1d1_l2d2p365_yearly_periodic(self):
        """Test with signal with len>365 and yearly_periodic set to True"""

        fname = "test_l2_l1d1_l2d2p365_data_input.csv"
        cvxpy_solver = "MOSEK"  # scs is only other option for tests as of 11/28/22

        # Test signal data; incl MOSEK and SCS expected solutions
        filepath = Path(__file__).parent.parent
        data_file_path = (
                filepath / "fixtures" / "signal_decompositions" / fname
        )
        test_data = pd.read_csv(data_file_path)

        # Raw signal
        signal = test_data["test_signal"].array
        # Expected output
        expected_s_hat = test_data[f"expected_s_hat_{cvxpy_solver.lower()}_yearly_periodic"].array
        expected_s_seas = test_data[f"expected_s_seas_{cvxpy_solver.lower()}_yearly_periodic"].array

        # Run test with default args
        actual_s_hat, actual_s_seas = sd.l2_l1d1_l2d2p365(
            signal,
            solver=cvxpy_solver,
            yearly_periodic=True
        )

        self.assertListAlmostEqual(list(expected_s_hat), list(actual_s_hat))
        self.assertListAlmostEqual(list(expected_s_seas), list(actual_s_seas))

    def test_l2_l1d1_l2d2p365_seas_max(self):
        """Test with signal with a max constraint on seas_max=0.5"""

        fname = "test_l2_l1d1_l2d2p365_data_input.csv"
        cvxpy_solver = "MOSEK"  # scs is only other option for tests as of 11/28/22

        # Test signal data; incl MOSEK and SCS expected solutions
        filepath = Path(__file__).parent.parent
        data_file_path = (
                filepath / "fixtures" / "signal_decompositions" / fname
        )
        test_data = pd.read_csv(data_file_path)

        # Raw signal
        signal = test_data["test_signal"].array[:365]
        # Expected output
        expected_s_hat = test_data[f"expected_s_hat_{cvxpy_solver.lower()}_seas_max_365"].array.dropna()
        expected_s_seas = test_data[f"expected_s_seas_{cvxpy_solver.lower()}_seas_max_365"].array.dropna()

        # Run test with default args
        actual_s_hat, actual_s_seas = sd.l2_l1d1_l2d2p365(
            signal,
            solver=cvxpy_solver,
            seas_max=0.5
        )

        self.assertListAlmostEqual(list(expected_s_hat), list(actual_s_hat))
        self.assertListAlmostEqual(list(expected_s_seas), list(actual_s_seas))

    # def test_l1_l2d2p365_default(self):
    #     """Test with default args"""
    #
    #     cvxpy_solver = "MOSEK"  # scs is only other option for tests as of 11/28/22
    #
    #     # Test signal data; incl MOSEK and SCS expected solutions
    #     filepath = Path(__file__).parent.parent
    #     data_file_path = (
    #             filepath / "fixtures" / "signal_decompositions" /
    #             "noisy_sim_data_signal_decomposition_input_365.csv"
    #     )
    #     test_data = pd.read_csv(data_file_path)
    #
    #     # Raw signal
    #     signal = test_data["test_signal"].array
    #     # Expected output
    #     expected_s_hat = test_data[f"expected_s_hat_{cvxpy_solver.lower()}"].array
    #     expected_s_seas = test_data[f"expected_s_seas_{cvxpy_solver.lower()}"].array
    #
    #     # Run test with default args
    #     actual_s_hat, actual_s_seas = sd.l1_l2d2p365(signal, solver=cvxpy_solver)
    #
    #     self.assertListAlmostEqual(list(expected_s_hat), list(actual_s_hat))
    #     self.assertListAlmostEqual(list(expected_s_seas), list(actual_s_seas))

    # def test_l1_l2d2p365_default_long(self):
    #     self.assertEqual(True, True)  # add assertion here
    # def test_l1_l2d2p365_idx_select(self):
    #     self.assertEqual(True, True)  # add assertion here
    #
    # def test_tl1_l2d2p365(self):
    #     self.assertEqual(True, False)  # add assertion here
    #
    # def test_tl1_l1d1_l2d2p365(self):
    #     self.assertEqual(True, False)  # add assertion here
    #
    # def test_make_l2_l1d2(self):
    #     self.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
    unittest.main()
