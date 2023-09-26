"""
!!!!!!!!! NOTE !!!!!!!!!!!!
These tests will fail and return a MOSEK error if a MOSEK license isn't found.
If required, please check that your license file is in [HOME]/mosek and is current.

This module contains tests for the following signal decompositions:

1) '_cvx_l2_l1d1_l2d2p365', components:
    - l2: gaussian noise, sum-of-squares small or l2-norm squared
    - l1d1: piecewise constant heuristic, l1-norm of first order differences
    - l2d2p365: small second order diffs (smooth) and 365-periodic

    TESTS
    -----
    - test_cvx_l2_l1d1_l2d2p365_default
    - test_cvx_l2_l1d1_l2d2p365_transition
    - test_cvx_l2_l1d1_l2d2p365_transition_wrong
    - test_cvx_l2_l1d1_l2d2p365_default_long
    - test_cvx_l2_l1d1_l2d2p365_idx_select
    - test_cvx_l2_l1d1_l2d2p365_yearly_periodic

2) '_cvx_tl1_l2d2p365', components:
    - tl1: 'tilted l1-norm,' also known as quantile cost function
    - l2d2p365: small second order diffs (smooth) and 365-periodic

    TESTS
    -----
    - test_cvx_tl1_l2d2p365_default
    - test_cvx_tl1_l2d2p365_idx_select
    - test_cvx_tl1_l2d2p365_long_not_yearly_periodic

3) '_cvx_l1_l1d1_l2d2p365', components:
    - l1: l1-norm
    - l1d1: piecewise constant heuristic, l1-norm of first order differences
    - l2d2p365: small second order diffs (smooth) and 365-periodic

    TESTS
    -----
    - test_cvx_l1_l1d1_l2d2p365_default
    - test_cvx_l1_l1d1_l2d2p365_idx_select

4) '_cvx_l2_l1d2_constrained':

    TESTS
    -----
    - test_cvx_l2_l1d2_constrained_default
"""

import unittest
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error as mae

from solardatatools import _cvx_signal_decompositions as sd


class TestSignalDecompositions(unittest.TestCase):
    def setUp(self):
        self.cvxpy_solver = "CLARABEL"  # all tests are using MOSEK
        self.mae_threshold = 0.003
        self.obj_tolerance = 1

    #######################
    # _cvx_l2_l1d1_l2d2p365
    #######################

    def test_cxv_l2_l1d1_l2d2p365_default(self):
        """Test with default args"""

        # Load input and output data
        filepath = Path(__file__).parent.parent
        data_file_path = (
            filepath
            / "fixtures"
            / "signal_decompositions"
            / "_cvx_signal_decompositions"
        )

        input_path = (
            str(data_file_path) + "/" + "test_cvx_l2_l1d1_l2d2p365_default_input.json"
        )
        output_path = (
            str(data_file_path) + "/" + "test_cvx_l2_l1d1_l2d2p365_default_output.json"
        )

        # Load input
        with open(input_path) as f:
            input = json.load(f)

        # Load output
        with open(output_path) as f:
            output = json.load(f)

        # Raw signal
        signal = np.array(input["test_signal"])

        # Expected output
        expected_s_hat = output["expected_s_hat_mosek_365"]
        expected_s_seas = output["expected_s_seas_mosek_365"]
        expected_obj_val = output["expected_obj_val_mosek_365"]

        # Run test
        actual_s_hat, actual_s_seas, _, actual_obj_val = sd._cvx_l2_l1d1_l2d2p365(
            signal, w1=10, w2=1e5, solver=self.cvxpy_solver, return_all=True
        )

        mae_s_hat = mae(actual_s_hat, expected_s_hat)
        mae_s_seas = mae(actual_s_seas, expected_s_seas)

        self.assertLess(mae_s_hat, self.mae_threshold)
        self.assertLess(mae_s_seas, self.mae_threshold)
        self.assertAlmostEqual(expected_obj_val, actual_obj_val, self.obj_tolerance)

    def test_cvx_l2_l1d1_l2d2p365_transition(self):
        """Test with piecewise fn transition location"""

        # Load input and output data
        filepath = Path(__file__).parent.parent
        data_file_path = (
            filepath
            / "fixtures"
            / "signal_decompositions"
            / "_cvx_signal_decompositions"
        )

        input_path = (
            str(data_file_path)
            + "/"
            + "test_cvx_l2_l1d1_l2d2p365_transition_input.json"
        )
        output_path = (
            str(data_file_path)
            + "/"
            + "test_cvx_l2_l1d1_l2d2p365_transition_output.json"
        )

        # Load input
        with open(input_path) as f:
            input = json.load(f)

        # Load output
        with open(output_path) as f:
            output = json.load(f)

        # Input
        signal = np.array(input["test_signal"])
        indices = input["indices"]

        # Expected output
        expected_s_hat = output["expected_s_hat_mosek_transition_365"]
        expected_s_seas = output["expected_s_seas_mosek_transition_365"]
        expected_obj_val = output["expected_obj_val_mosek_transition_365"]

        # Run test
        actual_s_hat, actual_s_seas, _, actual_obj_val = sd._cvx_l2_l1d1_l2d2p365(
            signal,
            w1=10,
            w2=1e5,
            transition_locs=indices,
            solver=self.cvxpy_solver,
            return_all=True,
        )

        mae_s_hat = mae(actual_s_hat, expected_s_hat)
        mae_s_seas = mae(actual_s_seas, expected_s_seas)

        self.assertLess(mae_s_hat, self.mae_threshold)
        self.assertLess(mae_s_seas, self.mae_threshold)
        self.assertAlmostEqual(expected_obj_val, actual_obj_val, self.obj_tolerance)

    def test_cvx_l2_l1d1_l2d2p365_transition_wrong(self):
        """Test with wrong (random) transition location"""

        # Load input and output data
        filepath = Path(__file__).parent.parent
        data_file_path = (
            filepath
            / "fixtures"
            / "signal_decompositions"
            / "_cvx_signal_decompositions"
        )

        input_path = (
            str(data_file_path)
            + "/"
            + "test_cvx_l2_l1d1_l2d2p365_transition_wrong_input.json"
        )
        output_path = (
            str(data_file_path)
            + "/"
            + "test_cvx_l2_l1d1_l2d2p365_transition_wrong_output.json"
        )

        # Load input
        with open(input_path) as f:
            input = json.load(f)

        # Load output
        with open(output_path) as f:
            output = json.load(f)

        # Input
        signal = np.array(input["test_signal"])
        transition = input["indices"]

        # Expected output
        expected_s_hat = output["expected_s_hat_mosek_transition_wrong_365"]
        expected_s_seas = output["expected_s_seas_mosek_transition_wrong_365"]
        expected_obj_val = output["expected_obj_val_mosek_transition_wrong_365"]

        # Run test
        actual_s_hat, actual_s_seas, _, actual_obj_val = sd._cvx_l2_l1d1_l2d2p365(
            signal,
            w1=10,
            w2=1e5,
            transition_locs=transition,
            solver=self.cvxpy_solver,
            return_all=True,
        )

        mae_s_hat = mae(actual_s_hat, expected_s_hat)
        mae_s_seas = mae(actual_s_seas, expected_s_seas)

        self.assertLess(mae_s_hat, self.mae_threshold)
        self.assertLess(mae_s_seas, self.mae_threshold)
        self.assertAlmostEqual(expected_obj_val, actual_obj_val, self.obj_tolerance)

    def test_cvx_l2_l1d1_l2d2p365_default_long(self):
        """Test with default args and signal with len >365"""

        # Load input and output data
        filepath = Path(__file__).parent.parent
        data_file_path = (
            filepath
            / "fixtures"
            / "signal_decompositions"
            / "_cvx_signal_decompositions"
        )

        input_path = (
            str(data_file_path)
            + "/"
            + "test_cvx_l2_l1d1_l2d2p365_default_long_input.json"
        )
        output_path = (
            str(data_file_path)
            + "/"
            + "test_cvx_l2_l1d1_l2d2p365_default_long_output.json"
        )

        # Load input
        with open(input_path) as f:
            input = json.load(f)

        # Load output
        with open(output_path) as f:
            output = json.load(f)

        # Raw signal
        signal = np.array(input["test_signal"])

        # Expected output
        expected_s_hat = output["expected_s_hat_mosek"]
        expected_s_seas = output["expected_s_seas_mosek"]
        expected_obj_val = output["expected_obj_val_mosek"]

        # Run test
        actual_s_hat, actual_s_seas, _, actual_obj_val = sd._cvx_l2_l1d1_l2d2p365(
            signal, w1=10, w2=1e5, solver=self.cvxpy_solver, return_all=True
        )

        mae_s_hat = mae(actual_s_hat, expected_s_hat)
        mae_s_seas = mae(actual_s_seas, expected_s_seas)

        self.assertLess(mae_s_hat, self.mae_threshold)
        self.assertLess(mae_s_seas, self.mae_threshold)
        self.assertAlmostEqual(expected_obj_val, actual_obj_val, self.obj_tolerance)

    def test_cvx_l2_l1d1_l2d2p365_idx_select(self):
        """Test with signal with select indices"""

        # Load input and output data
        filepath = Path(__file__).parent.parent
        data_file_path = (
            filepath
            / "fixtures"
            / "signal_decompositions"
            / "_cvx_signal_decompositions"
        )

        input_path = (
            str(data_file_path)
            + "/"
            + "test_cvx_l2_l1d1_l2d2p365_idx_select_input.json"
        )
        output_path = (
            str(data_file_path)
            + "/"
            + "test_cvx_l2_l1d1_l2d2p365_idx_select_output.json"
        )

        # Load input
        with open(input_path) as f:
            input = json.load(f)

        # Load output
        with open(output_path) as f:
            output = json.load(f)

        # Raw signal
        signal = np.array(input["test_signal"])
        indices = input["indices"]

        # Expected output
        expected_s_hat = output["expected_s_hat_mosek_ixs"]
        expected_s_seas = output["expected_s_seas_mosek_ixs"]
        expected_obj_val = output["expected_obj_val_mosek_ixs"]

        # Run test
        actual_s_hat, actual_s_seas, _, actual_obj_val = sd._cvx_l2_l1d1_l2d2p365(
            signal,
            w1=10,
            w2=1e5,
            solver=self.cvxpy_solver,
            use_ixs=indices,
            return_all=True,
        )

        mae_s_hat = mae(actual_s_hat, expected_s_hat)
        mae_s_seas = mae(actual_s_seas, expected_s_seas)

        self.assertLess(mae_s_hat, self.mae_threshold)
        self.assertLess(mae_s_seas, self.mae_threshold)
        self.assertAlmostEqual(expected_obj_val, actual_obj_val, self.obj_tolerance)

    def test_cvx_l2_l1d1_l2d2p365_yearly_periodic(self):
        """Test with signal with len>365 and yearly_periodic set to True"""

        # Load input and output data
        filepath = Path(__file__).parent.parent
        data_file_path = (
            filepath
            / "fixtures"
            / "signal_decompositions"
            / "_cvx_signal_decompositions"
        )

        input_path = (
            str(data_file_path)
            + "/"
            + "test_cvx_l2_l1d1_l2d2p365_yearly_periodic_input.json"
        )
        output_path = (
            str(data_file_path)
            + "/"
            + "test_cvx_l2_l1d1_l2d2p365_yearly_periodic_output.json"
        )

        # Load input
        with open(input_path) as f:
            input = json.load(f)

        # Load output
        with open(output_path) as f:
            output = json.load(f)

        # Raw signal
        signal = np.array(input["test_signal"])

        # Expected output
        expected_s_hat = output["expected_s_hat_mosek_yearly_periodic"]
        expected_s_seas = output["expected_s_seas_mosek_yearly_periodic"]
        expected_obj_val = output["expected_obj_val_mosek_yearly_periodic"]

        # Run test
        actual_s_hat, actual_s_seas, _, actual_obj_val = sd._cvx_l2_l1d1_l2d2p365(
            signal,
            w1=10,
            w2=1e5,
            solver=self.cvxpy_solver,
            yearly_periodic=True,
            return_all=True,
        )

        mae_s_hat = mae(actual_s_hat, expected_s_hat)
        mae_s_seas = mae(actual_s_seas, expected_s_seas)

        self.assertLess(mae_s_hat, self.mae_threshold)
        self.assertLess(mae_s_seas, self.mae_threshold)
        self.assertAlmostEqual(expected_obj_val, actual_obj_val, self.obj_tolerance)

    ###################
    # _cvx_tl1_l2d2p365
    ###################

    def test_cvx_tl1_l2d2p365_default(self):
        """Test with default args"""

        # Load input and output data
        filepath = Path(__file__).parent.parent
        data_file_path = (
            filepath
            / "fixtures"
            / "signal_decompositions"
            / "_cvx_signal_decompositions"
        )

        input_path = (
            str(data_file_path) + "/" + "test_cvx_tl1_l2d2p365_default_input.json"
        )
        output_path = (
            str(data_file_path) + "/" + "test_cvx_tl1_l2d2p365_default_output.json"
        )

        # Load input
        with open(input_path) as f:
            input = json.load(f)

        # Load output
        with open(output_path) as f:
            output = json.load(f)

        # Input
        signal = np.array(input["test_signal"])

        # Expected output
        expected_s_seas = output["expected_s_seas_mosek_365"]
        expected_obj_val = output["expected_obj_val_mosek_365"]

        # Run test with default args
        actual_s_seas, actual_obj_val = sd._cvx_tl1_l2d2p365(
            signal, tau=0.8, w1=1e5, solver=self.cvxpy_solver, return_all=True
        )

        mae_s_seas = mae(actual_s_seas, expected_s_seas)

        self.assertLess(mae_s_seas, self.mae_threshold)
        self.assertAlmostEqual(expected_obj_val, actual_obj_val, self.obj_tolerance)

    def test_cvx_tl1_l2d2p365_idx_select(self):
        """Test with select indices"""

        # Load input and output data
        filepath = Path(__file__).parent.parent
        data_file_path = (
            filepath
            / "fixtures"
            / "signal_decompositions"
            / "_cvx_signal_decompositions"
        )

        input_path = (
            str(data_file_path) + "/" + "test_cvx_tl1_l2d2p365_idx_select_input.json"
        )
        output_path = (
            str(data_file_path) + "/" + "test_cvx_tl1_l2d2p365_idx_select_output.json"
        )

        # Load input
        with open(input_path) as f:
            input = json.load(f)

        # Load output
        with open(output_path) as f:
            output = json.load(f)

        # Input
        signal = np.array(input["test_signal"])
        indices = input["indices"]

        # Expected output
        expected_s_seas = output["expected_s_seas_mosek_ixs"]
        expected_obj_val = output["expected_obj_val_mosek_ixs"]

        # Run test
        actual_s_seas, actual_obj_val = sd._cvx_tl1_l2d2p365(
            signal,
            tau=0.8,
            w1=1e5,
            solver=self.cvxpy_solver,
            use_ixs=indices,
            return_all=True,
        )

        mae_s_seas = mae(actual_s_seas, expected_s_seas)

        self.assertLess(mae_s_seas, self.mae_threshold)
        self.assertAlmostEqual(expected_obj_val, actual_obj_val, self.obj_tolerance)

    def test_cxv_tl1_l2d2p365_long_not_yearly_periodic(self):
        """Test with signal with len>365 and yearly_periodic set to True"""

        # Load input and output data
        filepath = Path(__file__).parent.parent
        data_file_path = (
            filepath
            / "fixtures"
            / "signal_decompositions"
            / "_cvx_signal_decompositions"
        )

        input_path = (
            str(data_file_path)
            + "/"
            + "test_cvx_tl1_l2d2p365_long_not_yearly_periodic_input.json"
        )
        output_path = (
            str(data_file_path)
            + "/"
            + "test_cvx_tl1_l2d2p365_long_not_yearly_periodic_output.json"
        )

        # Load input
        with open(input_path) as f:
            input = json.load(f)

        # Load output
        with open(output_path) as f:
            output = json.load(f)

        # Raw signal
        signal = np.array(input["test_signal"])

        # Expected output
        expected_s_seas = output["expected_s_seas_mosek_yearly_periodic"]
        expected_obj_val = output["expected_obj_val_mosek_yearly_periodic"]

        # Run test with default args
        actual_s_seas, actual_obj_val = sd._cvx_tl1_l2d2p365(
            signal,
            tau=0.8,
            solver=self.cvxpy_solver,
            w1=1e5,
            yearly_periodic=False,
            return_all=True,
        )

        mae_s_seas = mae(actual_s_seas, expected_s_seas)

        self.assertLess(mae_s_seas, self.mae_threshold)
        self.assertAlmostEqual(expected_obj_val, actual_obj_val, self.obj_tolerance)

    #######################
    # _cvx_l1_l1d1_l2d2p365
    #######################

    def test_cvx_l1_l1d1_l2d2p365_default(self):
        """Test with default args"""

        # Load input and output data
        filepath = Path(__file__).parent.parent
        data_file_path = (
            filepath
            / "fixtures"
            / "signal_decompositions"
            / "_cvx_signal_decompositions"
        )

        input_path = (
            str(data_file_path) + "/" + "test_cvx_l1_l1d1_l2d2p365_default_input.json"
        )
        output_path = (
            str(data_file_path) + "/" + "test_cvx_l1_l1d1_l2d2p365_default_output.json"
        )

        # Load input
        with open(input_path) as f:
            input = json.load(f)

        # Load output
        with open(output_path) as f:
            output = json.load(f)

        # Input
        signal = np.array(input["test_signal"])

        # Expected output
        expected_s_hat = output[f"expected_s_hat_mosek_365"]
        expected_s_seas = output[f"expected_s_seas_mosek_365"]
        expected_obj_val = output[f"expected_obj_val_mosek_365"]

        # Run test with default args
        actual_s_hat, actual_s_seas, _, actual_obj_val = sd._cvx_l1_l1d1_l2d2p365(
            signal, w1=5, w2=1e5, solver=self.cvxpy_solver, return_all=True
        )

        mae_s_hat = mae(actual_s_hat, expected_s_hat)
        mae_s_seas = mae(actual_s_seas, expected_s_seas)

        self.assertLess(mae_s_hat, self.mae_threshold)
        self.assertLess(mae_s_seas, self.mae_threshold)
        self.assertAlmostEqual(expected_obj_val, actual_obj_val, self.obj_tolerance)

    def test_cvx_l1_l1d1_l2d2p365_idx_select(self):
        """Test with select indices"""

        # Load input and output data
        filepath = Path(__file__).parent.parent
        data_file_path = (
            filepath
            / "fixtures"
            / "signal_decompositions"
            / "_cvx_signal_decompositions"
        )

        input_path = (
            str(data_file_path)
            + "/"
            + "test_cvx_l1_l1d1_l2d2p365_idx_select_input.json"
        )
        output_path = (
            str(data_file_path)
            + "/"
            + "test_cvx_l1_l1d1_l2d2p365_idx_select_output.json"
        )

        # Load input
        with open(input_path) as f:
            input = json.load(f)

        # Load output
        with open(output_path) as f:
            output = json.load(f)

        # Input
        signal = np.array(input["test_signal"])
        indices = input["indices"]

        # Expected output
        expected_s_hat = output[f"expected_s_hat_mosek_ixs"]
        expected_s_seas = output[f"expected_s_seas_mosek_ixs"]
        expected_obj_val = output[f"expected_obj_val_mosek_ixs"]

        # Run test
        actual_s_hat, actual_s_seas, _, actual_obj_val = sd._cvx_l1_l1d1_l2d2p365(
            signal,
            w1=5,
            w2=1e5,
            solver=self.cvxpy_solver,
            use_ixs=indices,
            return_all=True,
        )

        mae_s_hat = mae(actual_s_hat, expected_s_hat)
        mae_s_seas = mae(actual_s_seas, expected_s_seas)

        self.assertLess(mae_s_hat, self.mae_threshold)
        self.assertLess(mae_s_seas, self.mae_threshold)
        self.assertAlmostEqual(expected_obj_val, actual_obj_val, self.obj_tolerance)

        ###############################
        # _cvx_l2_l1d2_constrained
        ###############################

    def test_cvx_l2_l1d2_constrained_default(self):
        """Test with default args"""

        # Load input and output data
        filepath = Path(__file__).parent.parent
        data_file_path = (
            filepath
            / "fixtures"
            / "signal_decompositions"
            / "_cvx_signal_decompositions"
        )

        input_path = (
            str(data_file_path)
            + "/"
            + "test_cvx_l2_l1d2_constrained_default_input.json"
        )
        output_path = (
            str(data_file_path)
            + "/"
            + "test_cvx_l2_l1d2_constrained_default_output.json"
        )

        # Load input
        with open(input_path) as f:
            input = json.load(f)

        # Load output
        with open(output_path) as f:
            output = json.load(f)

        # Input
        signal = np.array(input["test_signal"])

        # Expected output
        expected_y_hat = output[f"expected_y_hat_mosek"]
        expected_obj_val = output[f"expected_obj_val_mosek"]

        # Run test with default args
        actual_y_hat, actual_obj_val = sd._cvx_l2_l1d2_constrained(
            signal, w1=1e1, solver=self.cvxpy_solver, return_all=True
        )

        mae_y_hat = mae(actual_y_hat, expected_y_hat)

        self.assertLess(mae_y_hat, self.mae_threshold)
        self.assertAlmostEqual(expected_obj_val, actual_obj_val, self.obj_tolerance)


if __name__ == "__main__":
    unittest.main()
