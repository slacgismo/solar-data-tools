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
    - test_l1_l2d2p365_long_not_yearly_periodic

3) 'tl1_l2d2p365', components:
    - tl1: 'tilted l1-norm,' also known as quantile cost function
    - l2d2p365: small second order diffs (smooth) and 365-periodic

    TESTS
    -----
    - test_tl1_l2d2p365_default
    - test_tl1_l2d2p365_idx_select
    - test_tl1_l2d2p365_long_not_yearly_periodic

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
import json
import numpy as np
from solardatatools import signal_decompositions as sd


class TestSignalDecompositions(unittest.TestCase):

    def setUp(self):
        self.cvxpy_solver = "MOSEK" # all tests are using MOSEK

    # TODO: can use np.testing.asset_array_almost_equal(l1, l2)
    # Tolerance for difference between solutions
    # to be updated for obj_value based on experiments
    tolerance = 5  # higher tolerance will fail w/ this rounded data
    def assertListAlmostEqual(self, list1, list2, tol=tolerance):
        self.assertEqual(len(list1), len(list2))
        for a, b in zip(list1, list2):
            self.assertAlmostEqual(a, b, tol)

    ##################
    # l2_l1d1_l2d2p365
    ##################

    def test_l2_l1d1_l2d2p365_default(self):
        """Test with default args"""

        # Load input and output data
        filepath = Path(__file__).parent.parent
        data_file_path = (filepath / "fixtures" / "signal_decompositions")

        input_path = str(data_file_path) + "/" + "test_l2_l1d1_l2d2p365_default_input.json"
        output_path = str(data_file_path) + "/" + "test_l2_l1d1_l2d2p365_default_output.json"

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
        c1 = 2 # adjusted weight to get a reasonable decomposition

        actual_s_hat, actual_s_seas, _, actual_obj_val = sd.l2_l1d1_l2d2p365(
            signal,
            c1=c1,
            solver=self.cvxpy_solver,
            return_obj=True
        )

        self.assertListAlmostEqual(list(expected_s_hat), list(actual_s_hat))
        self.assertListAlmostEqual(list(expected_s_seas), list(actual_s_seas))
        self.assertAlmostEqual(expected_obj_val, actual_obj_val)

    def test_l2_l1d1_l2d2p365_tv_weights(self):
        """Test with TV weights"""

        # Load input and output data
        filepath = Path(__file__).parent.parent
        data_file_path = (filepath / "fixtures" / "signal_decompositions")

        input_path = str(data_file_path) + "/" + "test_l2_l1d1_l2d2p365_tv_weights_input.json"
        output_path = str(data_file_path) + "/" + "test_l2_l1d1_l2d2p365_tv_weights_output.json"

        # Load input
        with open(input_path) as f:
            input = json.load(f)

        # Load output
        with open(output_path) as f:
            output = json.load(f)

        # Input
        signal = np.array(input["test_signal"])
        rand_tv_weights = np.array(input["rand_tv_weights_365"])

        # Expected output
        expected_s_hat = output["expected_s_hat_mosek_tvw_365"]
        expected_s_seas = output["expected_s_seas_mosek_tvw_365"]
        expected_obj_val = output["expected_obj_val_mosek_tvw_365"]

        # Run test
        c1 = 2.5 # adjusted weight to get a reasonable decomposition

        actual_s_hat, actual_s_seas, _, actual_obj_val = sd.l2_l1d1_l2d2p365(
            signal,
            c1=c1,
            solver=self.cvxpy_solver,
            tv_weights=rand_tv_weights,
            return_obj=True
        )

        self.assertListAlmostEqual(list(expected_s_hat), list(actual_s_hat))
        self.assertListAlmostEqual(list(expected_s_seas), list(actual_s_seas))
        self.assertAlmostEqual(expected_obj_val, actual_obj_val)

    def test_l2_l1d1_l2d2p365_transition(self):
        """Test with piecewise fn transition location"""

        # Load input and output data
        filepath = Path(__file__).parent.parent
        data_file_path = (filepath / "fixtures" / "signal_decompositions")

        input_path = str(data_file_path) + "/" + "test_l2_l1d1_l2d2p365_transition_input.json"
        output_path = str(data_file_path) + "/" + "test_l2_l1d1_l2d2p365_transition_output.json"

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
        actual_s_hat, actual_s_seas, _, actual_obj_val = sd.l2_l1d1_l2d2p365(
            signal,
            transition_locs=indices,
            return_obj=True
        )

        self.assertListAlmostEqual(list(expected_s_hat), list(actual_s_hat))
        self.assertListAlmostEqual(list(expected_s_seas), list(actual_s_seas))
        self.assertAlmostEqual(expected_obj_val, actual_obj_val)

    def test_l2_l1d1_l2d2p365_transition_wrong(self):
        """Test with wrong (random) transition location"""

        # Load input and output data
        filepath = Path(__file__).parent.parent
        data_file_path = (filepath / "fixtures" / "signal_decompositions")

        input_path = str(data_file_path) + "/" + "test_l2_l1d1_l2d2p365_transition_wrong_input.json"
        output_path = str(data_file_path) + "/" + "test_l2_l1d1_l2d2p365_transition_wrong_output.json"

        # Load input
        with open(input_path) as f:
            input = json.load(f)

        # Load output
        with open(output_path) as f:
            output = json.load(f)

        # Input
        signal = np.array(input["test_signal"])
        transition  = input["indices"]

        # Expected output
        expected_s_hat = output["expected_s_hat_mosek_transition_wrong_365"]
        expected_s_seas = output["expected_s_seas_mosek_transition_wrong_365"]
        expected_obj_val = output["expected_obj_val_mosek_transition_wrong_365"]

        # Run test
        actual_s_hat, actual_s_seas, _, actual_obj_val = sd.l2_l1d1_l2d2p365(
            signal,
            transition_locs=transition,
            return_obj=True
        )

        self.assertListAlmostEqual(list(expected_s_hat), list(actual_s_hat))
        self.assertListAlmostEqual(list(expected_s_seas), list(actual_s_seas))
        self.assertAlmostEqual(expected_obj_val, actual_obj_val)

    def test_l2_l1d1_l2d2p365_default_long(self):
        """Test with default args and signal with len >365"""

        # Load input and output data
        filepath = Path(__file__).parent.parent
        data_file_path = (filepath / "fixtures" / "signal_decompositions")

        input_path = str(data_file_path) + "/" + "test_l2_l1d1_l2d2p365_default_long_input.json"
        output_path = str(data_file_path) + "/" + "test_l2_l1d1_l2d2p365_default_long_output.json"

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
        c1 = 2 # adjusted weight to get a reasonable decomposition

        actual_s_hat, actual_s_seas, _, actual_obj_val = sd.l2_l1d1_l2d2p365(
            signal,
            c1=c1,
            solver=self.cvxpy_solver,
            return_obj=True
        )

        self.assertListAlmostEqual(list(expected_s_hat), list(actual_s_hat))
        self.assertListAlmostEqual(list(expected_s_seas), list(actual_s_seas))
        self.assertAlmostEqual(expected_obj_val, actual_obj_val)

    def test_l2_l1d1_l2d2p365_idx_select(self):
        """Test with signal with select indices"""

        # Load input and output data
        filepath = Path(__file__).parent.parent
        data_file_path = (filepath / "fixtures" / "signal_decompositions")

        input_path = str(data_file_path) + "/" + "test_l2_l1d1_l2d2p365_idx_select_input.json"
        output_path = str(data_file_path) + "/" + "test_l2_l1d1_l2d2p365_idx_select_output.json"

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
        c1 = 2 # adjusted weight to get a reasonable decomposition

        actual_s_hat, actual_s_seas, _, actual_obj_val = sd.l2_l1d1_l2d2p365(
            signal,
            c1=c1,
            solver=self.cvxpy_solver,
            use_ixs=indices,
            return_obj=True
        )

        self.assertListAlmostEqual(list(expected_s_hat), list(actual_s_hat))
        self.assertListAlmostEqual(list(expected_s_seas), list(actual_s_seas))
        self.assertAlmostEqual(expected_obj_val, actual_obj_val)

    def test_l2_l1d1_l2d2p365_yearly_periodic(self):
        """Test with signal with len>365 and yearly_periodic set to True"""

        # Load input and output data
        filepath = Path(__file__).parent.parent
        data_file_path = (filepath / "fixtures" / "signal_decompositions")

        input_path = str(data_file_path) + "/" + "test_l2_l1d1_l2d2p365_yearly_periodic_input.json"
        output_path = str(data_file_path) + "/" + "test_l2_l1d1_l2d2p365_yearly_periodic_output.json"

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
        c1 = 1 # adjusted weight to get a reasonable decomposition

        actual_s_hat, actual_s_seas, _, actual_obj_val = sd.l2_l1d1_l2d2p365(
            signal,
            c1=c1,
            solver=self.cvxpy_solver,
            yearly_periodic=True,
            return_obj=True
        )

        self.assertListAlmostEqual(list(expected_s_hat), list(actual_s_hat))
        self.assertListAlmostEqual(list(expected_s_seas), list(actual_s_seas))
        self.assertAlmostEqual(expected_obj_val, actual_obj_val)


    ##################
    # l1_l2d2p365
    ##################

    def test_l1_l2d2p365_default(self):
        """Test with default args"""

        # Load input and output data
        filepath = Path(__file__).parent.parent
        data_file_path = (filepath / "fixtures" / "signal_decompositions")

        input_path = str(data_file_path) + "/" + "test_l1_l2d2p365_default_input.json"
        output_path = str(data_file_path) + "/" + "test_l1_l2d2p365_default_output.json"

        # Load input
        with open(input_path) as f:
            input = json.load(f)

        # Load output
        with open(output_path) as f:
            output = json.load(f)

        # Raw signal
        signal = np.array(input["test_signal"])

        # Expected output
        expected_s_seas = output["expected_s_seas_mosek_365"]
        expected_obj_val = output["expected_obj_val_mosek_365"]

        # Run test with default args
        actual_s_seas, actual_obj_val = sd.l1_l2d2p365(signal, solver=self.cvxpy_solver, return_obj=True)

        self.assertListAlmostEqual(list(expected_s_seas), list(actual_s_seas))
        self.assertAlmostEqual(expected_obj_val, actual_obj_val)

    def test_l1_l2d2p365_idx_select(self):
        """Test with select indices"""

        # Load input and output data
        filepath = Path(__file__).parent.parent
        data_file_path = (filepath / "fixtures" / "signal_decompositions")

        input_path = str(data_file_path) + "/" + "test_l1_l2d2p365_idx_select_input.json"
        output_path = str(data_file_path) + "/" + "test_l1_l2d2p365_idx_select_output.json"

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
        actual_s_seas, actual_obj_val = sd.l1_l2d2p365(
            signal,
            solver=self.cvxpy_solver,
            use_ixs=indices,
            return_obj=True
        )

        self.assertListAlmostEqual(list(expected_s_seas), list(actual_s_seas))
        self.assertAlmostEqual(expected_obj_val, actual_obj_val)

    def test_l1_l2d2p365_long_not_yearly_periodic(self):
        """Test with signal with len>365 and yearly_periodic set to True"""

        # Load input and output data
        filepath = Path(__file__).parent.parent
        data_file_path = (filepath / "fixtures" / "signal_decompositions")

        input_path = str(data_file_path) + "/" + "test_l1_l2d2p365_long_not_yearly_periodic_input.json"
        output_path = str(data_file_path) + "/" + "test_l1_l2d2p365_long_not_yearly_periodic_output.json"

        # Load input
        with open(input_path) as f:
            input = json.load(f)

        # Load output
        with open(output_path) as f:
            output = json.load(f)

        # Input
        signal = np.array(input["test_signal"])

        # Expected output
        expected_s_seas = output["expected_s_seas_mosek_yearly_periodic"]
        expected_obj_val = output["expected_obj_val_mosek_yearly_periodic"]

        # Run test with default args
        actual_s_seas, actual_obj_val = sd.l1_l2d2p365(
          signal,
          solver=self.cvxpy_solver,
          yearly_periodic=False,
          return_obj=True
         )

        self.assertListAlmostEqual(list(expected_s_seas), list(actual_s_seas))
        self.assertAlmostEqual(expected_obj_val, actual_obj_val)


    ##############
    # tl1_l2d2p365
    ##############

    def test_tl1_l2d2p365_default(self):
        """Test with default args"""

        # Load input and output data
        filepath = Path(__file__).parent.parent
        data_file_path = (filepath / "fixtures" / "signal_decompositions")

        input_path = str(data_file_path) + "/" + "test_tl1_l2d2p365_default_input.json"
        output_path = str(data_file_path) + "/" + "test_tl1_l2d2p365_default_output.json"

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
        actual_s_seas, actual_obj_val = sd.tl1_l2d2p365(signal,
                                                        tau=0.8,
                                                        solver=self.cvxpy_solver,
                                                        return_obj=True)

        self.assertListAlmostEqual(list(expected_s_seas), list(actual_s_seas))
        self.assertAlmostEqual(expected_obj_val, actual_obj_val)

    def test_tl1_l2d2p365_idx_select(self):
        """Test with select indices"""

        # Load input and output data
        filepath = Path(__file__).parent.parent
        data_file_path = (filepath / "fixtures" / "signal_decompositions")

        input_path = str(data_file_path) + "/" + "test_tl1_l2d2p365_idx_select_input.json"
        output_path = str(data_file_path) + "/" + "test_tl1_l2d2p365_idx_select_output.json"

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
        actual_s_seas, actual_obj_val = sd.tl1_l2d2p365(signal,
                                                        tau=0.8,
                                                        solver=self.cvxpy_solver,
                                                        use_ixs=indices,
                                                        return_obj=True)

        self.assertListAlmostEqual(list(expected_s_seas), list(actual_s_seas))
        self.assertAlmostEqual(expected_obj_val, actual_obj_val)

    def test_tl1_l2d2p365_long_not_yearly_periodic(self):
        """Test with signal with len>365 and yearly_periodic set to True"""

        # Load input and output data
        filepath = Path(__file__).parent.parent
        data_file_path = (filepath / "fixtures" / "signal_decompositions")

        input_path = str(data_file_path) + "/" + "test_tl1_l2d2p365_long_not_yearly_periodic_input.json"
        output_path = str(data_file_path) + "/" + "test_tl1_l2d2p365_long_not_yearly_periodic_output.json"

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
        actual_s_seas, actual_obj_val = sd.tl1_l2d2p365(signal,
                                                        tau=0.8,
                                                        solver=self.cvxpy_solver,
                                                        yearly_periodic=False,
                                                        return_obj=True)

        self.assertListAlmostEqual(list(expected_s_seas), list(actual_s_seas))
        self.assertAlmostEqual(expected_obj_val, actual_obj_val)


    ###################
    # tl1_l1d1_l2d2p365
    ###################

    def test_tl1_l1d1_l2d2p365_default(self):
        """Test with default args"""

        # Load input and output data
        filepath = Path(__file__).parent.parent
        data_file_path = (filepath / "fixtures" / "signal_decompositions")

        input_path = str(data_file_path) + "/" + "test_tl1_l1d1_l2d2p365_default_input.json"
        output_path = str(data_file_path) + "/" + "test_tl1_l1d1_l2d2p365_default_output.json"

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
        actual_s_hat, actual_s_seas, _, _, actual_obj_val = sd.tl1_l1d1_l2d2p365(
            signal,
            tau=0.8,
            c1=5,
            c2=500,
            c3=100,
            solver=self.cvxpy_solver,
            return_obj=True
        )

        self.assertListAlmostEqual(list(expected_s_hat), list(actual_s_hat))
        self.assertListAlmostEqual(list(expected_s_seas), list(actual_s_seas))
        self.assertAlmostEqual(expected_obj_val, actual_obj_val)

    def test_tl1_l1d1_l2d2p365_idx_select(self):
        """Test with select indices"""

        # Load input and output data
        filepath = Path(__file__).parent.parent
        data_file_path = (filepath / "fixtures" / "signal_decompositions")

        input_path = str(data_file_path) + "/" + "test_tl1_l1d1_l2d2p365_idx_select_input.json"
        output_path = str(data_file_path) + "/" + "test_tl1_l1d1_l2d2p365_idx_select_output.json"

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
        actual_s_hat, actual_s_seas, _, _, actual_obj_val = sd.tl1_l1d1_l2d2p365(
            signal,
            tau=0.8,
            c1=5,
            c2=500,
            c3=100,
            solver=self.cvxpy_solver,
            use_ixs=indices,
            return_obj = True
        )

        self.assertListAlmostEqual(list(expected_s_hat), list(actual_s_hat))
        self.assertListAlmostEqual(list(expected_s_seas), list(actual_s_seas))
        self.assertAlmostEqual(expected_obj_val, actual_obj_val)

    def test_tl1_l1d1_l2d2p365_tv_weights(self):
        """Test with TV weights"""

        # Load input and output data
        filepath = Path(__file__).parent.parent
        data_file_path = (filepath / "fixtures" / "signal_decompositions")

        input_path = str(data_file_path) + "/" + "test_tl1_l1d1_l2d2p365_tv_weights_input.json"
        output_path = str(data_file_path) + "/" + "test_tl1_l1d1_l2d2p365_tv_weights_output.json"

        # Load input
        with open(input_path) as f:
            input = json.load(f)

        # Load output
        with open(output_path) as f:
            output = json.load(f)

        # Input
        signal = np.array(input["test_signal"])
        rand_tv_weights = np.array(input["rand_tv_weights_365"])

        # Expected output
        expected_s_hat = output[f"expected_s_hat_mosek_tvw_365"]
        expected_s_seas = output[f"expected_s_seas_mosek_tvw_365"]
        expected_obj_val = output[f"expected_obj_val_mosek_tvw_365"]

        # Run test
        actual_s_hat, actual_s_seas, _, _, actual_obj_val = sd.tl1_l1d1_l2d2p365(
            signal,
            tau=0.8,
            c1=5,
            c2=500,
            c3=100,
            solver=self.cvxpy_solver,
            tv_weights=rand_tv_weights,
            return_obj = True
        )

        self.assertListAlmostEqual(list(expected_s_hat), list(actual_s_hat))
        self.assertListAlmostEqual(list(expected_s_seas), list(actual_s_seas))
        self.assertAlmostEqual(expected_obj_val, actual_obj_val)


if __name__ == '__main__':
    unittest.main()
