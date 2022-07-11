import unittest
import numpy as np
# import os
from statistical_clear_sky.algorithm.initialization.linearization_helper\
 import LinearizationHelper

class TestLinealizationHelper(unittest.TestCase):
    '''
    Unit test for obtaining initial data of Right Vectors component r0,
    which is used as a denomoniator of non-linear equation in order to make
    it linear.
    It convers the first part of the constructor of main.IterativeClearSky
    in the original code.
    '''

    def setUp(self):
        pass

    def test_obtain_component_r0(self):

        power_signals_d = np.array([[3.65099996e-01, 0.00000000e+00,
                                     0.00000000e+00, 2.59570003e+00],
                                    [6.21100008e-01, 0.00000000e+00,
                                     0.00000000e+00, 2.67740011e+00],
                                    [8.12500000e-01, 0.00000000e+00,
                                     0.00000000e+00, 2.72729993e+00],
                                    [9.00399983e-01, 0.00000000e+00,
                                     0.00000000e+00, 2.77419996e+00]])
        rank_k = 4

        expected_result = np.array([1.36527916, 2.70624333, 4.04720749,
                                    5.38817165])

        linearization_helper = LinearizationHelper(solver_type='ECOS')
        left_low_rank_matrix_u, singular_values_sigma, right_low_rank_matrix_v \
            = np.linalg.svd(power_signals_d)
        initial_r_cs_value = np.diag(singular_values_sigma[:rank_k]).dot(
                    right_low_rank_matrix_v[:rank_k, :])
        actual_result = linearization_helper.obtain_component_r0(
            initial_r_cs_value)

        np.testing.assert_almost_equal(actual_result, expected_result,
                                       decimal=2)
