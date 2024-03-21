""" This module contains tests for the following signal decompositions:

1) 'l2_l1d1_l2d2p365', components:
    - l2: gaussian noise, sum-of-squares small or l2-norm squared
    - l1d1: piecewise constant heuristic, l1-norm of first order differences
    - l2d2p365: small second order diffs (smooth) and 365-periodic

    TESTS
    -----
    - test_l2_l1d1_l2d2p365_default
    - test_l2_l1d1_l2d2p365_default_long
    - test_l2_l1d1_l2d2p365_idx_select
    - test_l2_l1d1_l2d2p365_yearly_periodic
    - test_l2_l1d1_l2d2p365_yearly_periodic_sum_card
    - test_l2_l1d1_l2d2p365_osqp

2) 'tl1_l2d2p365', components:
    - tl1: 'tilted l1-norm,' also known as quantile cost function
    - l2d2p365: small second order diffs (smooth) and 365-periodic

    TESTS
    -----
    - test_tl1_l2d2p365_default
    - test_tl1_l2d2p365_idx_select
    - test_tl1_l2d2p365_long_not_yearly_periodic

3) 'l1_l1d1_l2d2p365', components:
    - l1: l1-norm
    - l1d1: piecewise constant heuristic, l1-norm of first order differences
    - l2d2p365: small second order diffs (smooth) and 365-periodic

    TESTS
    -----
    - test_l1_l1d1_l2d2p365_default
    - test_l1_l1d1_l2d2p365_idx_select
    - test_l1_l1d1_l2d2p365_osqp

4) 'l2_l1d2_constrained':

    TESTS
    -----
    - test_l2_l1d2_constrained_default
"""

import unittest
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error as mae

from solardatatools import signal_decompositions as sd


class TestSignalDecompositions(unittest.TestCase):
    def setUp(self):
        self.solver = "QSS"
        self.solver_convex = "OSQP"  # use OSQP for convex only problem
        self.mae_threshold = 0.001
        self.obj_tolerance = 1

    ##################
    # l2_l1d1_l2d2p365
    ##################

    def test_l2_l1d1_l2d2p365_default(self):
        """Test with default args"""

        # Load input and output data
        filepath = Path(__file__).parent.parent
        data_file_path = filepath / "fixtures" / "signal_decompositions"

        input_path = (
            str(data_file_path) + "/" + "test_l2_l1d1_l2d2p365_default_input.json"
        )
        output_path = (
            str(data_file_path) + "/" + "test_l2_l1d1_l2d2p365_default_output.json"
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
        expected_s_hat = output["expected_s_hat_365"]
        expected_s_seas = output["expected_s_seas_365"]
        expected_obj_val = output["expected_obj_val_365"]

        # Run test
        actual_s_hat, actual_s_seas, _, actual_problem = sd.l2_l1d1_l2d2p365(
            signal, w1=50, w2=1e6, solver=self.solver, return_all=True
        )

        actual_obj_val = actual_problem.objective_value

        mae_s_hat = mae(actual_s_hat, expected_s_hat)
        mae_s_seas = mae(actual_s_seas, expected_s_seas)

        self.assertLess(mae_s_hat, self.mae_threshold)
        self.assertLess(mae_s_seas, self.mae_threshold)
        self.assertAlmostEqual(expected_obj_val, actual_obj_val, self.obj_tolerance)

    def test_l2_l1d1_l2d2p365_default_long(self):
        """Test with default args and signal with len >365"""

        # Load input and output data
        filepath = Path(__file__).parent.parent
        data_file_path = filepath / "fixtures" / "signal_decompositions"

        input_path = (
            str(data_file_path) + "/" + "test_l2_l1d1_l2d2p365_default_long_input.json"
        )
        output_path = (
            str(data_file_path) + "/" + "test_l2_l1d1_l2d2p365_default_long_output.json"
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
        expected_s_hat = output["expected_s_hat"]
        expected_s_seas = output["expected_s_seas"]
        expected_obj_val = output["expected_obj_val"]

        # Run test
        actual_s_hat, actual_s_seas, _, actual_problem = sd.l2_l1d1_l2d2p365(
            signal, w1=50, w2=1e6, solver=self.solver, return_all=True
        )

        actual_obj_val = actual_problem.objective_value

        mae_s_hat = mae(actual_s_hat, expected_s_hat)
        mae_s_seas = mae(actual_s_seas, expected_s_seas)

        self.assertLess(mae_s_hat, self.mae_threshold)
        self.assertLess(mae_s_seas, self.mae_threshold)
        self.assertAlmostEqual(expected_obj_val, actual_obj_val, self.obj_tolerance)

    def test_l2_l1d1_l2d2p365_idx_select(self):
        """Test with signal with select indices"""

        # Load input and output data
        filepath = Path(__file__).parent.parent
        data_file_path = filepath / "fixtures" / "signal_decompositions"

        input_path = (
            str(data_file_path) + "/" + "test_l2_l1d1_l2d2p365_idx_select_input.json"
        )
        output_path = (
            str(data_file_path) + "/" + "test_l2_l1d1_l2d2p365_idx_select_output.json"
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
        expected_s_hat = output["expected_s_hat_ixs"]
        expected_s_seas = output["expected_s_seas_ixs"]
        expected_obj_val = output["expected_obj_val_ixs"]

        # Run test
        actual_s_hat, actual_s_seas, _, actual_problem = sd.l2_l1d1_l2d2p365(
            signal, w1=50, w2=1e6, solver=self.solver, use_ixs=indices, return_all=True
        )

        actual_obj_val = actual_problem.objective_value

        mae_s_hat = mae(actual_s_hat, expected_s_hat)
        mae_s_seas = mae(actual_s_seas, expected_s_seas)

        self.assertLess(mae_s_hat, self.mae_threshold)
        self.assertLess(mae_s_seas, self.mae_threshold)
        self.assertAlmostEqual(expected_obj_val, actual_obj_val, self.obj_tolerance)

    def test_l2_l1d1_l2d2p365_yearly_periodic(self):
        """Test with signal with len>365 and yearly_periodic set to True"""

        # Load input and output data
        filepath = Path(__file__).parent.parent
        data_file_path = filepath / "fixtures" / "signal_decompositions"

        input_path = (
            str(data_file_path)
            + "/"
            + "test_l2_l1d1_l2d2p365_yearly_periodic_input.json"
        )
        output_path = (
            str(data_file_path)
            + "/"
            + "test_l2_l1d1_l2d2p365_yearly_periodic_output.json"
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
        expected_s_hat = output["expected_s_hat_yearly_periodic"]
        expected_s_seas = output["expected_s_seas_yearly_periodic"]
        expected_obj_val = output["expected_obj_val_yearly_periodic"]

        # Run test
        actual_s_hat, actual_s_seas, _, actual_problem = sd.l2_l1d1_l2d2p365(
            signal,
            w1=50,
            w2=1e6,
            solver=self.solver,
            yearly_periodic=True,
            return_all=True,
        )

        actual_obj_val = actual_problem.objective_value

        mae_s_hat = mae(actual_s_hat, expected_s_hat)
        mae_s_seas = mae(actual_s_seas, expected_s_seas)

        self.assertLess(mae_s_hat, self.mae_threshold)
        self.assertLess(mae_s_seas, self.mae_threshold)
        self.assertAlmostEqual(expected_obj_val, actual_obj_val, self.obj_tolerance)

    def test_l2_l1d1_l2d2p365_yearly_periodic_sum_card(self):
        """Test with signal with len>365 and yearly_periodic and sum_card set to True"""

        # Load input and output data
        filepath = Path(__file__).parent.parent
        data_file_path = filepath / "fixtures" / "signal_decompositions"

        input_path = (
            str(data_file_path)
            + "/"
            + "test_l2_l1d1_l2d2p365_yearly_periodic_sum_card_input.json"
        )
        output_path = (
            str(data_file_path)
            + "/"
            + "test_l2_l1d1_l2d2p365_yearly_periodic_sum_card_output.json"
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
        expected_s_hat = output["expected_s_hat_yearly_periodic_sum_card"]
        expected_s_seas = output["expected_s_seas_yearly_periodic_sum_card"]
        expected_obj_val = output["expected_obj_val_yearly_periodic_sum_card"]

        # Run test
        actual_s_hat, actual_s_seas, _, actual_problem = sd.l2_l1d1_l2d2p365(
            signal,
            w1=50,
            w2=1e6,
            solver=self.solver,
            yearly_periodic=True,
            return_all=True,
            sum_card=True,
        )

        actual_obj_val = actual_problem.objective_value

        mae_s_hat = mae(actual_s_hat, expected_s_hat)
        mae_s_seas = mae(actual_s_seas, expected_s_seas)

        self.assertLess(mae_s_hat, self.mae_threshold)
        self.assertLess(mae_s_seas, self.mae_threshold)
        self.assertAlmostEqual(expected_obj_val, actual_obj_val, self.obj_tolerance)

    def test_l2_l1d1_l2d2p365_osqp(self):
        """Test with signal with len>365 and yearly_periodic and sum_card set to True"""

        # Load input and output data
        filepath = Path(__file__).parent.parent
        data_file_path = filepath / "fixtures" / "signal_decompositions"

        input_path = str(data_file_path) + "/" + "test_l2_l1d1_l2d2p365_osqp_input.json"
        output_path = (
            str(data_file_path) + "/" + "test_l2_l1d1_l2d2p365_osqp_output.json"
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
        expected_s_hat = output["expected_s_hat_osqp"]
        expected_s_seas = output["expected_s_seas_osqp"]
        expected_obj_val = output["expected_obj_val_osqp"]

        # Run test
        actual_s_hat, actual_s_seas, _, actual_problem = sd.l2_l1d1_l2d2p365(
            signal, w1=50, w2=1e6, solver="OSQP", yearly_periodic=True, return_all=True
        )

        actual_obj_val = actual_problem.objective_value

        mae_s_hat = mae(actual_s_hat, expected_s_hat)
        mae_s_seas = mae(actual_s_seas, expected_s_seas)

        self.assertLess(mae_s_hat, self.mae_threshold)
        self.assertLess(mae_s_seas, self.mae_threshold)
        self.assertAlmostEqual(expected_obj_val, actual_obj_val, self.obj_tolerance)

    ##############
    # tl1_l2d2p365
    ##############

    def test_tl1_l2d2p365_default(self):
        """Test with default args"""

        # Load input and output data
        filepath = Path(__file__).parent.parent
        data_file_path = filepath / "fixtures" / "signal_decompositions"

        input_path = str(data_file_path) + "/" + "test_tl1_l2d2p365_default_input.json"
        output_path = (
            str(data_file_path) + "/" + "test_tl1_l2d2p365_default_output.json"
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
        expected_s_seas = output["expected_s_seas_365"]
        expected_obj_val = output["expected_obj_val_365"]

        # Run test with default args
        actual_s_seas, actual_problem = sd.tl1_l2d2p365(
            signal, tau=0.8, w1=1e5, solver=self.solver_convex, return_all=True
        )
        actual_obj_val = actual_problem.objective_value

        mae_s_seas = mae(actual_s_seas, expected_s_seas)

        self.assertLess(mae_s_seas, self.mae_threshold)
        self.assertAlmostEqual(expected_obj_val, actual_obj_val, self.obj_tolerance)

    def test_tl1_l2d2p365_idx_select(self):
        """Test with select indices"""

        # Load input and output data
        filepath = Path(__file__).parent.parent
        data_file_path = filepath / "fixtures" / "signal_decompositions"

        input_path = (
            str(data_file_path) + "/" + "test_tl1_l2d2p365_idx_select_input.json"
        )
        output_path = (
            str(data_file_path) + "/" + "test_tl1_l2d2p365_idx_select_output.json"
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
        expected_s_seas = output["expected_s_seas_ixs"]
        expected_obj_val = output["expected_obj_val_ixs"]

        # Run test
        actual_s_seas, actual_problem = sd.tl1_l2d2p365(
            signal,
            tau=0.8,
            w1=1e5,
            solver=self.solver_convex,
            use_ixs=indices,
            return_all=True,
        )
        actual_obj_val = actual_problem.objective_value

        mae_s_seas = mae(actual_s_seas, expected_s_seas)

        self.assertLess(mae_s_seas, self.mae_threshold)
        self.assertAlmostEqual(expected_obj_val, actual_obj_val, self.obj_tolerance)

    def test_tl1_l2d2p365_long_not_yearly_periodic(self):
        """Test with signal with len>365 and yearly_periodic set to True"""

        # Load input and output data
        filepath = Path(__file__).parent.parent
        data_file_path = filepath / "fixtures" / "signal_decompositions"

        input_path = (
            str(data_file_path)
            + "/"
            + "test_tl1_l2d2p365_long_not_yearly_periodic_input.json"
        )
        output_path = (
            str(data_file_path)
            + "/"
            + "test_tl1_l2d2p365_long_not_yearly_periodic_output.json"
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
        expected_s_seas = output["expected_s_seas_yearly_periodic"]
        expected_obj_val = output["expected_obj_val_yearly_periodic"]

        # Run test with default args
        actual_s_seas, actual_problem = sd.tl1_l2d2p365(
            signal,
            tau=0.8,
            solver=self.solver_convex,
            w1=1e5,
            yearly_periodic=False,
            return_all=True,
        )
        actual_obj_val = actual_problem.objective_value

        mae_s_seas = mae(actual_s_seas, expected_s_seas)

        self.assertLess(mae_s_seas, self.mae_threshold)
        self.assertAlmostEqual(expected_obj_val, actual_obj_val, self.obj_tolerance)

    ###################
    # l1_l1d1_l2d2p365
    ###################

    def test_l1_l1d1_l2d2p365_default(self):
        """Test with default args"""

        # Load input and output data
        filepath = Path(__file__).parent.parent
        data_file_path = filepath / "fixtures" / "signal_decompositions"

        input_path = (
            str(data_file_path) + "/" + "test_l1_l1d1_l2d2p365_default_input.json"
        )
        output_path = (
            str(data_file_path) + "/" + "test_l1_l1d1_l2d2p365_default_output.json"
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
        expected_s_hat = output[f"expected_s_hat_365"]
        expected_s_seas = output[f"expected_s_seas_365"]
        expected_obj_val = output[f"expected_obj_val_365"]

        # Run test with default args
        actual_s_hat, actual_s_seas, _, actual_problem = sd.l1_l1d1_l2d2p365(
            signal,
            w0=1,
            w1=2,
            w2=1e3,
            sum_card=True,
            solver="CLARABEL",
            return_all=True,
        )

        actual_obj_val = actual_problem.objective_value

        mae_s_hat = mae(actual_s_hat, expected_s_hat)
        mae_s_seas = mae(actual_s_seas, expected_s_seas)

        self.assertLess(mae_s_hat, self.mae_threshold)
        self.assertLess(mae_s_seas, self.mae_threshold)
        self.assertAlmostEqual(expected_obj_val, actual_obj_val, self.obj_tolerance)

        ##########################
        # l2_l1d2_constrained
        ##########################

    def test_l2_l1d2_default(self):
        """Test with default args"""

        # Load input and output data
        filepath = Path(__file__).parent.parent
        data_file_path = filepath / "fixtures" / "signal_decompositions"

        input_path = str(data_file_path) + "/" + "test_l2_l1d2_default_input.json"
        output_path = str(data_file_path) + "/" + "test_l2_l1d2_default_output.json"

        # Load input
        with open(input_path) as f:
            input = json.load(f)

        # Load output
        with open(output_path) as f:
            output = json.load(f)

        # Input
        signal = np.array(input["test_signal"])

        # Expected output
        expected_y_hat = output[f"expected_y_hat"]
        expected_obj_val = output[f"expected_obj_val"]

        # Run test with default args
        actual_y_hat, actual_problem = sd.l2_l1d2_constrained(
            signal, w1=5, solver=self.solver_convex, return_all=True
        )
        actual_obj_val = actual_problem.objective_value

        mae_y_hat = mae(actual_y_hat, expected_y_hat)

        self.assertLess(mae_y_hat, self.mae_threshold)
        self.assertAlmostEqual(expected_obj_val, actual_obj_val, self.obj_tolerance)


if __name__ == "__main__":
    unittest.main()
