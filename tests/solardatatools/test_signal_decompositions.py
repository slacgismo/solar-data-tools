import unittest
from pathlib import Path
import pandas as pd
import numpy as np
from solardatatools import signal_decompositions as sd

# change c1 and c2?
# pass weights x2 --> results same
# transition_locs is None --> results same
# yearly periodic F/T
# len(signal)>365
# pass seas_max (scalar)
class TestSignalDecompositions(unittest.TestCase):

    def test_l2_l1d1_l2d2p365_defaults(self):
        cvxpy_solver = "MOSEK"

        # Test signal data; incl MOSEK and SCS expected solutions
        filepath = Path(__file__).parent.parent
        data_file_path = (
            filepath / "fixtures" / "signal_decompositions" /
            "noisy_sim_data_signal_decomposition_input.csv"
        )
        test_data = pd.read_csv(
            data_file_path, parse_dates=True
        )
        # Raw signal
        signal = test_data["test_signal"].array
        # Expected output
        expected_s_hat = test_data["expected_s_hat_mosek"].array
        expected_s_seas = test_data["expected_s_seas_mosek"].array

        # Run test with default args
        actual_s_hat, actual_s_seas = sd.l2_l1d1_l2d2p365(signal, solver=cvxpy_solver)

        # use almost equal when writing values myself w/ certain precision?
        self.assertEqual(expected_s_hat, actual_s_hat)
        self.assertEqual(expected_s_seas, actual_s_seas)

    def test_l2_l1d1_l2d2p365_w_weights(self):
        cvxpy_solver = "MOSEK"

        # Test signal data; incl MOSEK and SCS expected solutions
        filepath = Path(__file__).parent.parent
        data_file_path = (
            filepath / "fixtures" / "signal_decompositions" /
            "noisy_sim_data_signal_decomposition_input.csv"
        )
        test_data = pd.read_csv(
            data_file_path, parse_dates=True
        )
        # Raw signal
        signal = test_data["test_signal"].array
        # Expected output
        expected_s_hat = test_data["expected_s_hat_mosek"].array
        expected_s_seas = test_data["expected_s_seas_mosek"].array

        # Run test
        rand_weights = np.random.uniform(10, 200, len(signal)-1)
        actual_s_hat, actual_s_seas = sd.l2_l1d1_l2d2p365(
            signal,
            solver=cvxpy_solver,
            tv_weights=rand_weights
        )

        # use almost equal when writing values myself w/ certain precision?
        self.assertEqual(expected_s_hat, actual_s_hat)
        self.assertEqual(expected_s_seas, actual_s_seas)

    def test_l2_l1d1_l2d2p365_transition(self):
        cvxpy_solver = "MOSEK"
        transition  = 100

        # Test signal data; incl MOSEK and SCS expected solutions
        filepath = Path(__file__).parent.parent
        data_file_path = (
            filepath / "fixtures" / "signal_decompositions" /
            "noisy_sim_data_signal_decomposition_input.csv"
        )
        test_data = pd.read_csv(
            data_file_path, parse_dates=True
        )
        # Raw signal
        signal = test_data["test_signal"].array
        # Expected output
        expected_s_hat = test_data["expected_s_hat_mosek"].array
        expected_s_seas = test_data["expected_s_seas_mosek"].array

        # Run test
        rand_weights = np.random.uniform(10, 200, len(signal)-1)
        actual_s_hat, actual_s_seas = sd.l2_l1d1_l2d2p365(
            signal,
            transition_locs=transition
        )

        # use almost equal when writing values myself w/ certain precision?
        self.assertEqual(expected_s_hat, actual_s_hat)
        self.assertEqual(expected_s_seas, actual_s_seas)

    # def test_l1_l2d2p365(self):
    #     self.assertEqual(True, False)  # add assertion here
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
