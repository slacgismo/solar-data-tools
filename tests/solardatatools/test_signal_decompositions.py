""" This module contains tests for the following signal decompositions:

1) 'l2_l1d1_l2d2p365', components:
    - l2: gaussian noise, sum-of-squares small or l2-norm squared
    - l1d1: piecewise constant heuristic, l1-norm of first order differences
    - l2d2p365: small second order diffs (smooth) and 365-periodic

    TESTS
    -----
    - test_l2_l1d1_l2d2p365_default
    - test_l2_l1d1_l2d2p365_tv_weights
    - test_l2_l1d1_l2d2p365_residual_weights
    - test_l2_l1d1_l2d2p365_transition
    - test_l2_l1d1_l2d2p365_transition_wrong
    - test_l2_l1d1_l2d2p365_default_long
    - test_l2_l1d1_l2d2p365_idx_select
    - test_l2_l1d1_l2d2p365_yearly_periodic
    - test_l2_l1d1_l2d2p365_seas_max

2) 'l1_l2d2p365', components:
    - l1: laplacian noise, sum-of-absolute values or l1-norm
    - l2d2p365: small second order diffs (smooth) and 365-periodic

    TESTS
    -----
    - test_l1_l2d2p365_default
    - test_l1_l2d2p365_idx_select
    - test_l1_l2d2p365_long_yearly_periodic

3) 'tl1_l2d2p365', components:
    - tl1: 'tilted l1-norm,' also known as quantile cost function
    - l2d2p365: small second order diffs (smooth) and 365-periodic

    TESTS
    -----
    - test_tl1_l2d2p365_default
    - test_tl1_l2d2p365_idx_select
    - test_tl1_l2d2p365_long_yearly_periodic

4) 'tl1_l1d1_l2d2p365', components:
    - tl1: 'tilted l1-norm,' also known as quantile cost function
    - l1d1: piecewise constant heuristic, l1-norm of first order differences
    - l2d2p365: small second order diffs (smooth) and 365-periodic

    TESTS
    -----
    - test_tl1_l1d1_l2d2p365_default
    - test_tl1_l1d1_l2d2p365_idx_select
    - test_tl1_l1d1_l2d2p365_tv_weights
    - test_tl1_l1d1_l2d2p365_residual_weights

"""

import unittest
from pathlib import Path
import pandas as pd
import numpy as np
from solardatatools import signal_decompositions as sd


class TestSignalDecompositions(unittest.TestCase):
    # Tolerance for difference between solutions
    tolerance = 6  # higher tolerance will fail w/ this rounded data

    def assertListAlmostEqual(self, list1, list2, tol=tolerance):
        self.assertEqual(len(list1), len(list2))
        for a, b in zip(list1, list2):
            self.assertAlmostEqual(a, b, tol)


    ##################
    # l2_l1d1_l2d2p365
    ##################

    def test_l2_l1d1_l2d2p365_default(self):
        """Test with default args"""

        fname = "test_l2_l1d1_l2d2p365_data_input.csv"
        cvxpy_solver = "MOSEK"
        c1 = 2 # adjusted weight to get a reasonable decomposition

        # Test signal data; incl MOSEK expected solutions
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
        expected_obj_val = test_data[f"expected_obj_val_{cvxpy_solver.lower()}_365"][0]

        # Run test
        actual_s_hat, actual_s_seas, actual_obj_val = sd.l2_l1d1_l2d2p365(
            signal,
            c1=c1,
            solver=cvxpy_solver,
            return_obj=True
        )

        self.assertListAlmostEqual(list(expected_s_hat), list(actual_s_hat))
        self.assertListAlmostEqual(list(expected_s_seas), list(actual_s_seas))
        self.assertAlmostEqual(expected_obj_val, actual_obj_val)

    def test_l2_l1d1_l2d2p365_tv_weights(self):
        """Test with TV weights"""

        fname = "test_l2_l1d1_l2d2p365_data_input.csv"
        cvxpy_solver = "MOSEK"
        c1 = 2.5 # adjusted weight to get a reasonable decomposition

        # Test signal data; incl MOSEK expected solutions
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
        expected_obj_val = test_data[f"expected_obj_val_{cvxpy_solver.lower()}_tvw_365"][0]

        # Run test
        rand_tv_weights = test_data["rand_tv_weights_365"].dropna() # len(signal)-1
        actual_s_hat, actual_s_seas, actual_obj_val = sd.l2_l1d1_l2d2p365(
            signal,
            c1=c1,
            solver=cvxpy_solver,
            tv_weights=rand_tv_weights,
            return_obj=True
        )

        self.assertListAlmostEqual(list(expected_s_hat), list(actual_s_hat))
        self.assertListAlmostEqual(list(expected_s_seas), list(actual_s_seas))
        self.assertAlmostEqual(expected_obj_val, actual_obj_val)

    def test_l2_l1d1_l2d2p365_residual_weights(self):
        """Test with residual weights"""

        fname = "test_l2_l1d1_l2d2p365_data_input.csv"
        cvxpy_solver = "MOSEK"
        c1 = 2.5 # adjusted weight to get a reasonable decomposition

        # Test signal data; incl MOSEK expected solutions
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
        expected_obj_val = test_data[f"expected_obj_val_{cvxpy_solver.lower()}_rw_365"][0]

        # Run test
        rand_residual_weights = test_data["rand_residual_weights_365"].dropna()
        actual_s_hat, actual_s_seas, actual_obj_val = sd.l2_l1d1_l2d2p365(
            signal,
            c1=c1,
            solver=cvxpy_solver,
            residual_weights=rand_residual_weights,
            return_obj=True
        )

        self.assertListAlmostEqual(list(expected_s_hat), list(actual_s_hat))
        self.assertListAlmostEqual(list(expected_s_seas), list(actual_s_seas))
        self.assertAlmostEqual(expected_obj_val, actual_obj_val)

    def test_l2_l1d1_l2d2p365_transition(self):
        """Test with piecewise fn transition location"""

        transition  = [133, 266]
        fname = "test_l2_l1d1_l2d2p365_data_input.csv"
        cvxpy_solver = "MOSEK"

        # Test signal data; incl MOSEK expected solutions
        filepath = Path(__file__).parent.parent
        data_file_path = (
                filepath / "fixtures" / "signal_decompositions" / fname
        )
        test_data = pd.read_csv(data_file_path)

        # Raw signal
        signal = test_data["test_signal"].array[:365]
        # Expected output
        expected_s_hat = test_data[f"expected_s_hat_{cvxpy_solver.lower()}_transition_365"].array.dropna()
        expected_s_seas = test_data[f"expected_s_seas_{cvxpy_solver.lower()}_transition_365"].array.dropna()
        expected_obj_val = test_data[f"expected_obj_val_{cvxpy_solver.lower()}_transition_365"][0]

        # Run test
        rand_weights = np.random.uniform(10, 200, len(signal)-1)
        actual_s_hat, actual_s_seas, actual_obj_val = sd.l2_l1d1_l2d2p365(
            signal,
            transition_locs=transition,
            return_obj=True
        )

        self.assertListAlmostEqual(list(expected_s_hat), list(actual_s_hat))
        self.assertListAlmostEqual(list(expected_s_seas), list(actual_s_seas))
        self.assertAlmostEqual(expected_obj_val, actual_obj_val)

    def test_l2_l1d1_l2d2p365_transition_wrong(self):
        """Test with wrong (random) transition location"""

        transition  = [100, 308]
        fname = "test_l2_l1d1_l2d2p365_data_input.csv"
        cvxpy_solver = "MOSEK"

        # Test signal data; incl MOSEK expected solutions
        filepath = Path(__file__).parent.parent
        data_file_path = (
                filepath / "fixtures" / "signal_decompositions" / fname
        )
        test_data = pd.read_csv(data_file_path)

        # Raw signal
        signal = test_data["test_signal"].array[:365]
        # Expected output
        expected_s_hat = test_data[f"expected_s_hat_{cvxpy_solver.lower()}_transition_wrong_365"].array.dropna()
        expected_s_seas = test_data[f"expected_s_seas_{cvxpy_solver.lower()}_transition_wrong_365"].array.dropna()
        expected_obj_val = test_data[f"expected_obj_val_{cvxpy_solver.lower()}_transition_wrong_365"][0]

        # Run test
        rand_weights = np.random.uniform(10, 200, len(signal)-1)
        actual_s_hat, actual_s_seas, actual_obj_val = sd.l2_l1d1_l2d2p365(
            signal,
            transition_locs=transition,
            return_obj=True
        )

        self.assertListAlmostEqual(list(expected_s_hat), list(actual_s_hat))
        self.assertListAlmostEqual(list(expected_s_seas), list(actual_s_seas))
        self.assertAlmostEqual(expected_obj_val, actual_obj_val)

    def test_l2_l1d1_l2d2p365_default_long(self):
        """Test with default args and signal with len >365"""

        fname = "test_l2_l1d1_l2d2p365_data_input.csv"
        cvxpy_solver = "MOSEK"
        c1 = 2 # adjusted weight to get a reasonable decomposition

        # Test signal data; incl MOSEK expected solutions
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
        expected_obj_val = test_data[f"expected_obj_val_{cvxpy_solver.lower()}"][0]

        # Run test
        actual_s_hat, actual_s_seas, actual_obj_val = sd.l2_l1d1_l2d2p365(
            signal,
            c1=c1,
            solver=cvxpy_solver,
            return_obj=True
        )

        self.assertListAlmostEqual(list(expected_s_hat), list(actual_s_hat))
        self.assertListAlmostEqual(list(expected_s_seas), list(actual_s_seas))
        self.assertAlmostEqual(expected_obj_val, actual_obj_val)

    def test_l2_l1d1_l2d2p365_idx_select(self):
        """Test with signal with select indices"""

        fname = "test_l2_l1d1_l2d2p365_data_input.csv"
        cvxpy_solver = "MOSEK"
        c1 = 2 # adjusted weight to get a reasonable decomposition

        # Test signal data; incl MOSEK expected solutions
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
        expected_obj_val = test_data[f"expected_obj_val_{cvxpy_solver.lower()}_ixs"][0]

        # Run test
        actual_s_hat, actual_s_seas, actual_obj_val = sd.l2_l1d1_l2d2p365(
            signal,
            c1=c1,
            solver=cvxpy_solver,
            use_ixs=indices,
            return_obj=True
        )

        self.assertListAlmostEqual(list(expected_s_hat), list(actual_s_hat))
        self.assertListAlmostEqual(list(expected_s_seas), list(actual_s_seas))
        self.assertAlmostEqual(expected_obj_val, actual_obj_val)

    def test_l2_l1d1_l2d2p365_yearly_periodic(self):
        """Test with signal with len>365 and yearly_periodic set to True"""

        fname = "test_l2_l1d1_l2d2p365_data_input.csv"
        cvxpy_solver = "MOSEK"
        c1 = 1 # adjusted weight to get a reasonable decomposition

        # Test signal data; incl MOSEK expected solutions
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
        expected_obj_val = test_data[f"expected_obj_val_{cvxpy_solver.lower()}_yearly_periodic"][0]

        # Run test
        actual_s_hat, actual_s_seas, actual_obj_val = sd.l2_l1d1_l2d2p365(
            signal,
            c1=c1,
            solver=cvxpy_solver,
            yearly_periodic=True,
            return_obj=True
        )

        self.assertListAlmostEqual(list(expected_s_hat), list(actual_s_hat))
        self.assertListAlmostEqual(list(expected_s_seas), list(actual_s_seas))
        self.assertAlmostEqual(expected_obj_val, actual_obj_val)

    def test_l2_l1d1_l2d2p365_seas_max(self):
        """Test with signal with a max constraint on seas_max=0.5"""

        fname = "test_l2_l1d1_l2d2p365_data_input.csv"
        cvxpy_solver = "MOSEK"

        # Test signal data; incl MOSEK expected solutions
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
        expected_obj_val = test_data[f"expected_obj_val_{cvxpy_solver.lower()}_seas_max_365"][0]

        # Run test
        actual_s_hat, actual_s_seas, actual_obj_val = sd.l2_l1d1_l2d2p365(
            signal,
            solver=cvxpy_solver,
            seas_max=0.5,
            return_obj=True
        )

        self.assertListAlmostEqual(list(expected_s_hat), list(actual_s_hat))
        self.assertListAlmostEqual(list(expected_s_seas), list(actual_s_seas))
        self.assertAlmostEqual(expected_obj_val, actual_obj_val)


    # ##################
    # # l1_l2d2p365
    # ##################
    #
    # def test_l1_l2d2p365_default(self):
    #     """Test with default args"""
    #
    #     fname = "test_l1_l2d2p365_data_input.csv"
    #     cvxpy_solver = "MOSEK"
    #     #c1 = 2 # adjusted weight to get a reasonable decomposition
    #
    #     # Test signal data; incl MOSEK expected solutions
    #     filepath = Path(__file__).parent.parent
    #     data_file_path = (
    #         filepath / "fixtures" / "signal_decompositions" / fname
    #     )
    #     test_data = pd.read_csv(data_file_path)
    #
    #     # Raw signal
    #     signal = test_data["test_signal"].array[:365]
    #     # Expected output
    #     expected_s_seas = test_data[f"expected_s_seas_{cvxpy_solver.lower()}_365"].array.dropna()
    #     expected_obj_val = test_data[f"expected_obj_val_{cvxpy_solver.lower()}_365"][0]
    #
    #     # Run test with default args
    #     actual_s_seas, actual_obj_val = sd.l1_l2d2p365(signal, solver=cvxpy_solver)
    #
    #     self.assertListAlmostEqual(list(expected_s_seas), list(actual_s_seas))
    #     self.assertAlmostEqual(expected_obj_val, actual_obj_val)
    #
    # def test_l1_l2d2p365_idx_select(self):
    #     """Test with select indices"""
    #
    #     fname = "test_l1_l2d2p365_data_input.csv"
    #     cvxpy_solver = "MOSEK"
    #     #c1 = 2 # adjusted weight to get a reasonable decomposition
    #
    #     # Test signal data; incl MOSEK expected solutions
    #     filepath = Path(__file__).parent.parent
    #     data_file_path = (
    #         filepath / "fixtures" / "signal_decompositions" / fname
    #     )
    #     test_data = pd.read_csv(data_file_path)
    #
    #     # Take first 300 days of dataset
    #     indices = list([True] * 300) + list([False] * (730 - 300))
    #
    #     # Raw signal
    #     signal = test_data["test_signal"].array
    #     # Expected output
    #     expected_s_seas = test_data[f"expected_s_seas_{cvxpy_solver.lower()}_ixs"].array.dropna()
    #     expected_obj_val = test_data[f"expected_obj_val_{cvxpy_solver.lower()}_ixs"][0]
    #
    #     # Run test
    #     actual_s_seas, actual_obj_val = sd.l1_l2d2p365(signal, solver=cvxpy_solver, use_ixs=indices, return_obj=True)
    #
    #     self.assertListAlmostEqual(list(expected_s_seas), list(actual_s_seas))
    #     self.assertAlmostEqual(expected_obj_val, actual_obj_val)
    #
    # def test_l1_l2d2p365_long_yearly_periodic(self):
    #     """Test with signal with len>365 and yearly_periodic set to True"""
    #
    #     fname = "test_l1_l2d2p365_data_input.csv"
    #     cvxpy_solver = "MOSEK"
    #     # c1 = 2 # adjusted weight to get a reasonable decomposition
    #
    #     # Test signal data; incl MOSEK expected solutions
    #     filepath = Path(__file__).parent.parent
    #     data_file_path = (
    #             filepath / "fixtures" / "signal_decompositions" / fname
    #     )
    #     test_data = pd.read_csv(data_file_path)
    #
    #     # Raw signal
    #     signal = test_data["test_signal"].array
    #     # Expected output
    #     expected_s_seas = test_data[f"expected_s_seas_{cvxpy_solver.lower()}_yearly_periodic"].array.dropna()
    #     expected_obj_val = test_data[f"expected_obj_val_{cvxpy_solver.lower()}_yearly_periodic"][0]
    #
    #     # Run test with default args
    #     actual_s_seas, actual_obj_val = sd.l2_l1d1_l2d2p365(
    #       signal,
    #       solver=cvxpy_solver,
    #       yearly_periodic=True,
    #       return_obj=True
    #      )
    #
    #     self.assertListAlmostEqual(list(expected_s_seas), list(actual_s_seas))
    #     self.assertAlmostEqual(expected_obj_val, actual_obj_val)
    #
    #
    # ##############
    # # tl1_l2d2p365
    # ##############
    #
    # def test_tl1_l2d2p365_default(self):
    #     """Test with default args"""
    #
    #     fname = "test_tl1_l2d2p365_data_input.csv"
    #     cvxpy_solver = "MOSEK"
    #     #c1 = 2 # adjusted weight to get a reasonable decomposition
    #
    #     # Test signal data; incl MOSEK expected solutions
    #     filepath = Path(__file__).parent.parent
    #     data_file_path = (
    #         filepath / "fixtures" / "signal_decompositions" / fname
    #     )
    #     test_data = pd.read_csv(data_file_path)
    #
    #     # Raw signal
    #     signal = test_data["test_signal"].array[:365]
    #     # Expected output
    #     expected_s_seas = test_data[f"expected_s_seas_{cvxpy_solver.lower()}_365"].array.dropna()
    #     expected_obj_val = test_data[f"expected_obj_val_{cvxpy_solver.lower()}_365"][0]
    #
    #     # Run test with default args
    #     actual_s_seas, actual_obj_val = sd.tl1_l2d2p365(signal, solver=cvxpy_solver, return_obj=True)
    #
    #     self.assertListAlmostEqual(list(expected_s_seas), list(actual_s_seas))
    #     self.assertAlmostEqual(expected_obj_val, actual_obj_val)
    #
    # def test_tl1_l2d2p365_idx_select(self):
    #     """Test with select indices"""
    #
    #     fname = "test_tl1_l2d2p365_data_input.csv"
    #     cvxpy_solver = "MOSEK"
    #     #c1 = 2 # adjusted weight to get a reasonable decomposition
    #
    #     # Test signal data; incl MOSEK expected solutions
    #     filepath = Path(__file__).parent.parent
    #     data_file_path = (
    #         filepath / "fixtures" / "signal_decompositions" / fname
    #     )
    #     test_data = pd.read_csv(data_file_path)
    #
    #     # Take first 300 days of dataset
    #     indices = list([True] * 300) + list([False] * (730 - 300))
    #
    #     # Raw signal
    #     signal = test_data["test_signal"].array
    #     # Expected output
    #     expected_s_seas = test_data[f"expected_s_seas_{cvxpy_solver.lower()}_ixs"].array.dropna()
    #     expected_obj_val = test_data[f"expected_obj_val_{cvxpy_solver.lower()}_ixs"][0]
    #
    #     # Run test
    #     actual_s_seas, actual_obj_val = sd.tl1_l2d2p365(signal, solver=cvxpy_solver, use_ixs=indices, return_obj=True)
    #
    #     self.assertListAlmostEqual(list(expected_s_seas), list(actual_s_seas))
    #     self.assertAlmostEqual(expected_obj_val, actual_obj_val)
    #
    # def test_tl1_l2d2p365_long_yearly_periodic(self):
    #     """Test with signal with len>365 and yearly_periodic set to True"""
    #
    #     fname = "test_tl1_l2d2p365_data_input.csv"
    #     cvxpy_solver = "MOSEK"
    #     # c1 = 2 # adjusted weight to get a reasonable decomposition
    #
    #     # Test signal data; incl MOSEK expected solutions
    #     filepath = Path(__file__).parent.parent
    #     data_file_path = (
    #             filepath / "fixtures" / "signal_decompositions" / fname
    #     )
    #     test_data = pd.read_csv(data_file_path)
    #
    #     # Raw signal
    #     signal = test_data["test_signal"].array
    #     # Expected output
    #     expected_s_seas = test_data[f"expected_s_seas_{cvxpy_solver.lower()}_yearly_periodic"].array.dropna()
    #     expected_obj_val = test_data[f"expected_obj_val_{cvxpy_solver.lower()}_yearly_periodic"][0]
    #
    #     # Run test with default args
    #     actual_s_seas, actual_obj_val = sd.tl1_l2d2p365(signal, solver=cvxpy_solver, yearly_periodic=True, return_obj=True)
    #
    #     self.assertListAlmostEqual(list(expected_s_seas), list(actual_s_seas))
    #     self.assertAlmostEqual(expected_obj_val, actual_obj_val)
    #
    #
    # ###################
    # # tl1_l1d1_l2d2p365
    # ###################
    #
    # def test_tl1_l2d2p365_default(self):
    #     """Test with default args"""
    #
    #     fname = "test_tl1_l1d1_l2d2p365_data_input.csv"
    #     cvxpy_solver = "MOSEK"
    #     # c1 = 2 # adjusted weight to get a reasonable decomposition
    #
    #     # Test signal data; incl MOSEK expected solutions
    #     filepath = Path(__file__).parent.parent
    #     data_file_path = (
    #             filepath / "fixtures" / "signal_decompositions" / fname
    #     )
    #     test_data = pd.read_csv(data_file_path)
    #
    #     # Raw signal
    #     signal = test_data["test_signal"].array[:365]
    #     # Expected output
    #     expected_s_hat = test_data[f"expected_s_hat_{cvxpy_solver.lower()}_365"].array.dropna()
    #     expected_s_seas = test_data[f"expected_s_seas_{cvxpy_solver.lower()}_365"].array.dropna()
    #     expected_obj_val = test_data[f"expected_obj_val_{cvxpy_solver.lower()}_365"][0]
    #
    #     # Run test with default args
    #     actual_s_hat, actual_s_seas, actual_obj_val = sd.tl1_l1d1_l2d2p365(signal, solver=cvxpy_solver, return_obj=True)
    #
    #     self.assertListAlmostEqual(list(expected_s_hat), list(actual_s_hat))
    #     self.assertListAlmostEqual(list(expected_s_seas), list(actual_s_seas))
    #     self.assertAlmostEqual(expected_obj_val, actual_obj_val)
    #
    # def test_tl1_l1d1_l2d2p365_idx_select(self):
    #     """Test with select indices"""
    #
    #     fname = "test_tl1_l2d2p365_data_input.csv"
    #     cvxpy_solver = "MOSEK"
    #     # c1 = 2 # adjusted weight to get a reasonable decomposition
    #
    #     # Test signal data; incl MOSEK expected solutions
    #     filepath = Path(__file__).parent.parent
    #     data_file_path = (
    #             filepath / "fixtures" / "signal_decompositions" / fname
    #     )
    #     test_data = pd.read_csv(data_file_path)
    #
    #     # Take first 300 days of dataset
    #     indices = list([True] * 300) + list([False] * (730 - 300))
    #
    #     # Raw signal
    #     signal = test_data["test_signal"].array
    #     # Expected output
    #     expected_s_hat = test_data[f"expected_s_hat_{cvxpy_solver.lower()}_idx"].array.dropna()
    #     expected_s_seas = test_data[f"expected_s_seas_{cvxpy_solver.lower()}_idx"].array.dropna()
    #     expected_obj_val = test_data[f"expected_obj_val_{cvxpy_solver.lower()}_idx"][0]
    #
    #     # Run test
    #     actual_s_hat, actual_s_seas, actual_obj_val = sd.tl1_l1d1_l2d2p365(
    #         signal,
    #         solver=cvxpy_solver,
    #         use_ixs=indices,
    #         return_obj = True
    #     )
    #
    #     self.assertListAlmostEqual(list(expected_s_hat), list(actual_s_hat))
    #     self.assertListAlmostEqual(list(expected_s_seas), list(actual_s_seas))
    #     self.assertAlmostEqual(expected_obj_val, actual_obj_val)
    #
    # def test_tl1_l1d1_l2d2p365_tv_weights(self):
    #     """Test with TV weights"""
    #
    #     fname = "test_tl1_l1d1_l2d2p365_data_input.csv"
    #     cvxpy_solver = "MOSEK"
    #
    #     # Test signal data; incl MOSEK expected solutions
    #     filepath = Path(__file__).parent.parent
    #     data_file_path = (
    #             filepath / "fixtures" / "signal_decompositions" / fname
    #     )
    #     test_data = pd.read_csv(data_file_path)
    #
    #     # Raw signal
    #     signal = test_data["test_signal"].array[:365]
    #     # Expected output
    #     expected_s_hat = test_data[f"expected_s_hat_{cvxpy_solver.lower()}_tvw_365"].array.dropna()
    #     expected_s_seas = test_data[f"expected_s_seas_{cvxpy_solver.lower()}_tvw_365"].array.dropna()
    #     expected_obj_val = test_data[f"expected_obj_val_{cvxpy_solver.lower()}_tvw_365"][0]
    #
    #     # Run test
    #     rand_tv_weights = test_data["rand_tv_weights_365"].dropna()  # len(signal)-1
    #     actual_s_hat, actual_s_seas, actual_obj_val = sd.tl1_l1d1_l2d2p365(
    #         signal,
    #         solver=cvxpy_solver,
    #         tv_weights=rand_tv_weights,
    #         return_obj = True
    #     )
    #
    #     self.assertListAlmostEqual(list(expected_s_hat), list(actual_s_hat))
    #     self.assertListAlmostEqual(list(expected_s_seas), list(actual_s_seas))
    #     self.assertAlmostEqual(expected_obj_val, actual_obj_val)
    #
    # def test_tl1_l1d1_l2d2p365_residual_weights(self):
    #     """Test with residual weights"""
    #
    #     fname = "test_tl1_l1d1_l2d2p365_data_input.csv"
    #     cvxpy_solver = "MOSEK"
    #
    #     # Test signal data; incl MOSEK expected solutions
    #     filepath = Path(__file__).parent.parent
    #     data_file_path = (
    #             filepath / "fixtures" / "signal_decompositions" / fname
    #     )
    #     test_data = pd.read_csv(data_file_path)
    #
    #     # Raw signal
    #     signal = test_data["test_signal"].array[:365]
    #     # Expected output
    #     expected_s_hat = test_data[f"expected_s_hat_{cvxpy_solver.lower()}_rw_365"].array.dropna()
    #     expected_s_seas = test_data[f"expected_s_seas_{cvxpy_solver.lower()}_rw_365"].array.dropna()
    #     expected_obj_val = test_data[f"expected_obj_val_{cvxpy_solver.lower()}_rw_365"][0]
    #
    #     # Run test
    #     rand_residual_weights = test_data["rand_residual_weights_365"].dropna()
    #     actual_s_hat, actual_s_seas, actual_obj_val = sd.tl1_l1d1_l2d2p365(
    #         signal,
    #         solver=cvxpy_solver,
    #         residual_weights=rand_residual_weights,
    #         return_obj = True
    #     )
    #
    #     self.assertListAlmostEqual(list(expected_s_hat), list(actual_s_hat))
    #     self.assertListAlmostEqual(list(expected_s_seas), list(actual_s_seas))
    #     self.assertAlmostEqual(expected_obj_val, actual_obj_val)

if __name__ == '__main__':
    unittest.main()
