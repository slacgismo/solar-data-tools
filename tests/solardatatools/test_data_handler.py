import unittest
from pathlib import Path
import pandas as pd
import numpy as np
from solardatatools import DataHandler
import matplotlib.pyplot as plt


class TestDataHandler(unittest.TestCase):

    def test_load_and_run(self):
        filepath = Path(__file__).parent.parent
        data_file_path = \
            filepath / 'fixtures' / 'data_transforms' / 'timeseries.csv'
        df = pd.read_csv(data_file_path, index_col=0, parse_dates=True)
        dh = DataHandler(df)
        dh.run_pipeline(verbose=False)
        # dh.report()
        self.assertAlmostEqual(
            dh.capacity_estimate, 6.7453649044036865, places=2
        )
        self.assertAlmostEqual(
            dh.data_quality_score, 0.9948186528497409, places=3
        )
        self.assertAlmostEqual(
            dh.data_clearness_score, 0.49222797927461137, places=3
        )
        self.assertTrue(dh.inverter_clipping)
        self.assertFalse(dh.time_shifts)
        scores_path = \
            filepath / 'fixtures' / 'scoring' / 'clipping_1.csv'
        with open(scores_path) as file:
            expected_scores = np.loadtxt(file, delimiter=',')
        np.testing.assert_array_almost_equal(
            dh.daily_scores.clipping_1, expected_scores, decimal=3
        )
        scores_path = \
            filepath / 'fixtures' / 'scoring' / 'clipping_2.csv'
        with open(scores_path) as file:
            expected_scores = np.loadtxt(file, delimiter=',')
        np.testing.assert_array_almost_equal(
            dh.daily_scores.clipping_2, expected_scores, decimal=3
        )
        scores_path = \
            filepath / 'fixtures' / 'scoring' / 'density.csv'
        with open(scores_path) as file:
            expected_scores = np.loadtxt(file, delimiter=',')
        np.testing.assert_array_almost_equal(
            dh.daily_scores.density, expected_scores, decimal=3
        )
        scores_path = \
            filepath / 'fixtures' / 'scoring' / 'linearity.csv'
        with open(scores_path) as file:
            expected_scores = np.loadtxt(file, delimiter=',')
        np.testing.assert_array_almost_equal(
            dh.daily_scores.linearity, expected_scores, decimal=2
        )
        scores_path = \
            filepath / 'fixtures' / 'scoring' / 'quality_clustering.csv'
        with open(scores_path) as file:
            expected_scores = np.loadtxt(file, delimiter=',')
        np.testing.assert_array_almost_equal(
            dh.daily_scores.quality_clustering, expected_scores, decimal=3
        )