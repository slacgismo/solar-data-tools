import unittest
import os
import numpy as np
import cvxpy as cvx
from statistical_clear_sky.algorithm.minimization.right_matrix\
 import RightMatrixMinimization

class TestRightMatrixMinimization(unittest.TestCase):

    def test_minimize_with_large_data(self):

        input_power_signals_file_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__),
            "../../fixtures/right_matrix_minimization",
            "three_years_power_signals_d_1.csv"))
        with open(input_power_signals_file_path) as file:
            power_signals_d = np.loadtxt(file, delimiter=',')

        rank_k = 6

        weights_file_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__),
                "../../fixtures/right_matrix_minimization",
                "three_years_weights.csv"))
        with open(weights_file_path) as file:
            weights = np.loadtxt(file, delimiter=',')

        tau = 0.9
        mu_r = 1e3

        initial_r0_value_file_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__),
            "../../fixtures/right_matrix_minimization",
            "three_years_initial_component_r0.csv"))
        with open(initial_r0_value_file_path) as file:
            initial_component_r0_value = np.loadtxt(file, delimiter=',')

        initial_l_cs_value_file_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__),
            "../../fixtures/right_matrix_minimization",
            "l_cs_value_after_left_matrix_minimization_iteration_1.csv"))
        with open(initial_l_cs_value_file_path) as file:
            initial_l_cs_value = np.loadtxt(file, delimiter=',')

        initial_r_cs_value_file_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__),
            "../../fixtures/right_matrix_minimization",
            "r_cs_value_after_left_matrix_minimization_iteration_1.csv"))
        with open(initial_r_cs_value_file_path) as file:
            initial_r_cs_value = np.loadtxt(file, delimiter=',')

        initial_beta_value = 0.0

        l_cs_value_after_iteration_1_file_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__),
            "../../fixtures/right_matrix_minimization",
            "l_cs_value_after_right_matrix_minimization_iteration_1.csv"))
        with open(l_cs_value_after_iteration_1_file_path) as file:
            expected_l_cs_value = np.loadtxt(file, delimiter=',')

        r_cs_value_after_iteration_1_file_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__),
            "../../fixtures/right_matrix_minimization",
            "r_cs_value_after_right_matrix_minimization_iteration_1.csv"))
        with open(r_cs_value_after_iteration_1_file_path) as file:
            expected_r_cs_value = np.loadtxt(file, delimiter=',')

        expected_beta_value = -0.04015762

        right_matrix_minimization = RightMatrixMinimization(power_signals_d,
            rank_k, weights, tau, mu_r, solver_type='MOSEK')

        try:
            actual_l_cs_value, actual_r_cs_value, actual_beta_value =\
                right_matrix_minimization.minimize(initial_l_cs_value,
                                                   initial_r_cs_value,
                                                   initial_beta_value,
                                                   initial_component_r0_value)
        except cvx.SolverError:
            self.skipTest("This test uses MOSEK solver"
                + "because default ECOS solver fails with large data. "
                + "Unless MOSEK is installed, this test fails.")
        else:
            np.testing.assert_array_almost_equal(
                actual_l_cs_value, expected_l_cs_value, decimal=2
            )
            np.testing.assert_array_almost_equal(
                actual_r_cs_value, expected_r_cs_value, decimal=1
            )
            np.testing.assert_almost_equal(
                actual_beta_value, expected_beta_value, decimal=4
            )
