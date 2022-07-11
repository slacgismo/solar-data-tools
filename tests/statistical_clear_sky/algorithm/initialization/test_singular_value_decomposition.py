import unittest
import numpy as np
from\
 statistical_clear_sky.algorithm.initialization.singular_value_decomposition\
 import SingularValueDecomposition

class TestSingularValueDecomposition(unittest.TestCase):

    def test_adjust_singular_vectors(self):

        power_signals_d = np.array([[3.65099996e-01, 0.00000000e+00,
                                     0.00000000e+00, 2.59570003e+00],
                                    [6.21100008e-01, 0.00000000e+00,
                                     0.00000000e+00, 2.67740011e+00],
                                    [8.12500000e-01, 0.00000000e+00,
                                     0.00000000e+00, 2.72729993e+00],
                                    [9.00399983e-01, 0.00000000e+00,
                                     0.00000000e+00, 2.77419996e+00]])

        left_singular_vectors_u = np.array([[0.46881027, -0.77474963,
                                            0.39354624, 0.1584339],
                                           [0.49437073, -0.15174524,
                                            -0.6766346, -0.52415321],
                                           [-0.51153077, 0.32155093,
                                            -0.27710787, 0.74709605],
                                           [-0.5235941, 0.52282062,
                                            0.55722365, -0.37684163]])
        right_singular_vectors_v = np.array([[0.24562222, 0.0, 0.0, 0.96936563],
                                            [0.96936563, 0.0, 0.0, -0.24562222],
                                            [0.0, 1.0, 0.0, 0.0],
                                            [0.0, 0.0, 1.0, 0.0]])

        expected_left_singular_vectors_u = np.array([[-0.46881027, -0.77474963,
                                                     0.39354624, 0.1584339],
                                                    [-0.49437073, -0.15174524,
                                                     -0.6766346, -0.52415321],
                                                    [0.51153077, 0.32155093,
                                                     -0.27710787, 0.74709605],
                                                    [0.5235941, 0.52282062,
                                                     0.55722365, -0.37684163]])
        expected_right_singular_vectors_v = np.array([[-0.24562222, 0.0,
                                                      0.0, -0.96936563],
                                                     [0.96936563, 0.0,
                                                      0.0, -0.24562222],
                                                     [0.0, 1.0, 0.0, 0.0],
                                                     [0.0, 0.0, 1.0, 0.0]])

        decomposition = SingularValueDecomposition()

        actual_left_singular_vectors_u, actual_right_singular_vectors_v = \
            decomposition._adjust_singular_vectors(
                left_singular_vectors_u, right_singular_vectors_v)

        np.testing.assert_array_equal(actual_left_singular_vectors_u,
                                      expected_left_singular_vectors_u)
        np.testing.assert_array_equal(actual_right_singular_vectors_v,
                                      expected_right_singular_vectors_v)
