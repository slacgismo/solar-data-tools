import unittest
import os
import numpy as np
from statistical_clear_sky.algorithm.iterative_fitting import IterativeFitting


class TestIterativeFitting(unittest.TestCase):
    def test_calculate_objective(self):

        input_power_signals_file_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "../fixtures/objective_calculation",
                "three_years_power_signals_d_1.csv",
            )
        )
        with open(input_power_signals_file_path) as file:
            power_signals_d = np.loadtxt(file, delimiter=",")

        rank_k = 6

        mu_l = 5e2
        mu_r = 1e3
        tau = 0.9

        initial_l_cs_value_file_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "../fixtures/objective_calculation",
                "three_years_initial_l_cs_value.csv",
            )
        )
        with open(initial_l_cs_value_file_path) as file:
            l_cs_value = np.loadtxt(file, delimiter=",")

        initial_r_cs_value_file_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "../fixtures/objective_calculation",
                "three_years_initial_r_cs_value.csv",
            )
        )
        with open(initial_r_cs_value_file_path) as file:
            r_cs_value = np.loadtxt(file, delimiter=",")

        beta_value = 0.0

        weights_file_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "../fixtures/objective_calculation",
                "three_years_weights.csv",
            )
        )
        with open(weights_file_path) as file:
            weights = np.loadtxt(file, delimiter=",")

        expected_objective_values = np.array(
            [
                117277.71151791142,
                478.8539994379723,
                23800125.708200675,
                228653.22102385858,
            ]
        )

        iterative_fitting = IterativeFitting(power_signals_d, rank_k=rank_k)

        actual_objective_values = iterative_fitting._calculate_objective(
            mu_l,
            mu_r,
            tau,
            l_cs_value,
            r_cs_value,
            beta_value,
            weights,
            sum_components=False,
        )

        # np.testing.assert_array_equal(actual_objective_values,
        #                               expected_objective_values)
        np.testing.assert_almost_equal(
            actual_objective_values, expected_objective_values, decimal=8
        )
