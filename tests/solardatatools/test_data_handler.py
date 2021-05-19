import unittest
import os
import pandas as pd
import numpy as np
from solardatatools import DataHandler

class TestDataHandler(unittest.TestCase):

    def test_load_and_run(self):
        data_file_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__),
                         "../fixtures/data_transforms/timeseries.csv"))
        df = pd.read_csv(data_file_path, index_col=0, parse_dates=True)
        dh = DataHandler(df)
        dh.run_pipeline()
        self.assertAlmostEqual(
            dh.data_quality_score, 0.9948186528497409, places=3
        )
        self.assertAlmostEqual(
            dh.data_clearness_score, 0.49222797927461137, places=3
        )
        self.assertTrue(dh.inverter_clipping)
        self.assertFalse(dh.time_shifts)
        scores_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__),
                         "../fixtures/density_scoring/density_scores.csv"))
        with open(scores_path) as file:
            expected_scores = np.loadtxt(file, delimiter=',')
        np.testing.assert_array_almost_equal(
            dh.daily_scores.density, expected_scores
        )