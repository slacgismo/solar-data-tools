import unittest
import numpy as np
import tempfile
import shutil
import os
from statistical_clear_sky.algorithm.iterative_fitting import IterativeFitting

class TestSerializationMixin(unittest.TestCase):

    def setUp(self):
        self._temp_directory = tempfile.mkdtemp()
        self._filepath = os.path.join(self._temp_directory, 'state_data.json')

    def tearDown(self):
        shutil.rmtree(self._temp_directory)

    def test_serialization(self):

        power_signals_d = np.array([[3.65099996e-01, 0.00000000e+00,
                                     0.00000000e+00, 2.59570003e+00],
                                    [6.21100008e-01, 0.00000000e+00,
                                     0.00000000e+00, 2.67740011e+00],
                                    [8.12500000e-01, 0.00000000e+00,
                                     0.00000000e+00, 2.72729993e+00],
                                    [9.00399983e-01, 0.00000000e+00,
                                     0.00000000e+00, 2.77419996e+00]])
        rank_k = 4

        original_iterative_fitting = IterativeFitting(power_signals_d,
                                                      rank_k=rank_k)

        original_iterative_fitting.save_instance(self._filepath)

        deserialized_iterative_fitting = IterativeFitting.load_instance(
            self._filepath)

        np.testing.assert_array_equal(deserialized_iterative_fitting.
                                      _power_signals_d,
                                      original_iterative_fitting.
                                      _power_signals_d)
