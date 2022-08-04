import unittest
from unittest.mock import Mock
import os
import numpy as np
import cvxpy as cvx
from statistical_clear_sky.algorithm.iterative_fitting import IterativeFitting
from statistical_clear_sky.algorithm.initialization.linearization_helper\
 import LinearizationHelper
from statistical_clear_sky.algorithm.initialization.weight_setting\
 import WeightSetting
from statistical_clear_sky.algorithm.minimization.left_matrix\
 import LeftMatrixMinimization
from statistical_clear_sky.algorithm.minimization.right_matrix\
 import RightMatrixMinimization

class TestIterativeFittingExecute(unittest.TestCase):

    def setUp(self):

        fixed_power_signals_d_file_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__),
            "../fixtures/for_mock/three_years_power_signals_d_1.csv"))
        with open(fixed_power_signals_d_file_path) as file:
            fixed_power_signals_d = np.loadtxt(file, delimiter=',')

        initial_r0_value_file_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__),
            "../fixtures/for_mock/three_years_initial_component_r0.csv"))
        with open(initial_r0_value_file_path) as file:
            linearization_helper_return_value = np.loadtxt(file, delimiter=',')

        self.mock_linearization_helper = Mock(spec=LinearizationHelper)
        self.mock_linearization_helper.obtain_component_r0.return_value =\
            linearization_helper_return_value

        weights_file_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__),
                         "../fixtures/for_mock/three_years_weights.csv"))
        with open(weights_file_path) as file:
            weight_setting_return_value = np.loadtxt(file, delimiter=',')

        self.mock_weight_setting = Mock(spec=WeightSetting)
        self.mock_weight_setting.obtain_weights.return_value =\
            weight_setting_return_value

        left_matrix_minimize_return_values = []

        for i in range(13):

            l_cs_value_left_matrix_file_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__),
                "../fixtures/for_mock",
                ("l_cs_value_after_left_matrix_minimization_iteration_{}.csv"
                 .format(i+1))))
            with open(l_cs_value_left_matrix_file_path) as file:
                l_cs_value_left_matrix = np.loadtxt(file, delimiter=',')
            r_cs_value_left_matrix_file_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__),
                "../fixtures/for_mock",
                ("r_cs_value_after_left_matrix_minimization_iteration_{}.csv"
                 .format(i+1))))
            with open(r_cs_value_left_matrix_file_path) as file:
                r_cs_value_left_matrix = np.loadtxt(file, delimiter=',')
            beta_value_left_matrix_file_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__),
                "../fixtures/for_mock",
                ("beta_value_after_left_matrix_minimization_iteration_{}.csv"
                 .format(i+1))))
            with open(beta_value_left_matrix_file_path) as file:
                beta_value_left_matrix = np.loadtxt(file, delimiter=',')

            left_matrix_minimize_return_values.append(
                (l_cs_value_left_matrix, r_cs_value_left_matrix,
                 beta_value_left_matrix))

        self.mock_left_matrix_minimization = Mock(spec=LeftMatrixMinimization)
        self.mock_left_matrix_minimization.minimize.side_effect =\
            left_matrix_minimize_return_values

        right_matrix_minimize_return_values = []

        for i in range(13):

            l_cs_value_right_matrix_file_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__),
                "../fixtures/for_mock",
                ("l_cs_value_after_right_matrix_minimization_iteration_{}.csv"
                 .format(i+1))))
            with open(l_cs_value_right_matrix_file_path) as file:
                l_cs_value_right_matrix = np.loadtxt(file, delimiter=',')
            r_cs_value_right_matrix_file_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__),
                "../fixtures/for_mock",
                ("r_cs_value_after_right_matrix_minimization_iteration_{}.csv"
                 .format(i+1))))
            with open(r_cs_value_right_matrix_file_path) as file:
                r_cs_value_right_matrix = np.loadtxt(file, delimiter=',')
            beta_value_right_matrix_file_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__),
                "../fixtures/for_mock",
                ("beta_value_after_right_matrix_minimization_iteration_{}.csv"
                 .format(i+1))))
            with open(beta_value_right_matrix_file_path) as file:
                beta_value_right_matrix = np.loadtxt(file, delimiter=',')

            right_matrix_minimize_return_values.append(
                (l_cs_value_right_matrix, r_cs_value_right_matrix,
                 beta_value_right_matrix))

        self.mock_right_matrix_minimization = Mock(spec=RightMatrixMinimization)
        self.mock_right_matrix_minimization.minimize.side_effect =\
            right_matrix_minimize_return_values

    def test_execute(self):

        input_power_signals_file_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__),
            "../fixtures/for_mock/three_years_power_signals_d_1.csv"))
        with open(input_power_signals_file_path) as file:
            power_signals_d = np.loadtxt(file, delimiter=',')

        rank_k = 6

        clear_sky_signals_file_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__),
                "../fixtures/for_mock/three_years_clear_sky_signals.csv"))
        with open(clear_sky_signals_file_path) as file:
            expected_clear_sky_signals = np.loadtxt(file, delimiter=',')
        expected_degradation_rate = np.array(-0.04069624)

        iterative_fitting = IterativeFitting(power_signals_d, rank_k=rank_k)

        # Inject mock objects by dependency injection:
        iterative_fitting.set_linearization_helper(
            self.mock_linearization_helper)
        iterative_fitting.set_weight_setting(self.mock_weight_setting)
        iterative_fitting.set_left_matrix_minimization(
            self.mock_left_matrix_minimization)
        iterative_fitting.set_right_matrix_minimization(
            self.mock_right_matrix_minimization)

        iterative_fitting.execute(mu_l=5e2, mu_r=1e3, tau=0.9,
                                  max_iteration=15, verbose=False)

        actual_clear_sky_signals = iterative_fitting.clear_sky_signals()
        actual_degradation_rate = iterative_fitting.degradation_rate()

        # Note: Discrepancy is due to the difference in Python 3.6 and 3.7.
        # np.testing.assert_array_equal(actual_clear_sky_signals,
        #                               expected_clear_sky_signals)
        np.testing.assert_almost_equal(actual_clear_sky_signals,
                                       expected_clear_sky_signals,
                                       decimal=13)
        # np.testing.assert_array_equal(actual_degradation_rate,
        #                               expected_degradation_rate)
        np.testing.assert_almost_equal(actual_degradation_rate,
                                       expected_degradation_rate,
                                       decimal=8)
