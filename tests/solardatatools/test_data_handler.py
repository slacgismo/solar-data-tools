import unittest
from pathlib import Path
import pandas as pd
import numpy as np
from solardatatools import DataHandler
import matplotlib.pyplot as plt


class TestDataHandler(unittest.TestCase):
    def test_load_and_run(self):
        filepath = Path(__file__).parent.parent
        data_file_path = filepath / "fixtures" / "data_transforms" / "timeseries.csv"
        df = pd.read_csv(data_file_path, parse_dates=[0], index_col=0)
        dh = DataHandler(df)
        dh.fix_dst()
        dh.run_pipeline(power_col="ac_power_01", fix_shifts=True, verbose=False)
        # dh.report()
        self.assertAlmostEqual(dh.capacity_estimate, 6.7453649044036865, places=2)
        self.assertAlmostEqual(dh.data_quality_score, 0.9948186528497409, places=3)
        self.assertAlmostEqual(dh.data_clearness_score, 0.49222797927461137, places=3)
        self.assertTrue(dh.inverter_clipping)
        self.assertFalse(dh.time_shifts)
        scores_path = filepath / "fixtures" / "scoring" / "clipping_1.csv"
        with open(scores_path) as file:
            expected_scores = np.loadtxt(file, delimiter=",")
        np.testing.assert_allclose(
            dh.daily_scores.clipping_1, expected_scores, atol=1e-3
        )
        scores_path = filepath / "fixtures" / "scoring" / "clipping_2.csv"
        with open(scores_path) as file:
            expected_scores = np.loadtxt(file, delimiter=",")
        np.testing.assert_allclose(
            dh.daily_scores.clipping_2, expected_scores, atol=2e-3
        )
        scores_path = filepath / "fixtures" / "scoring" / "density.csv"
        with open(scores_path) as file:
            expected_scores = np.loadtxt(file, delimiter=",")
        np.testing.assert_allclose(dh.daily_scores.density, expected_scores, atol=1e-3)
        scores_path = filepath / "fixtures" / "scoring" / "linearity.csv"
        with open(scores_path) as file:
            expected_scores = np.loadtxt(file, delimiter=",")
        np.testing.assert_allclose(
            dh.daily_scores.linearity, expected_scores, atol=2e-2
        )
        scores_path = filepath / "fixtures" / "scoring" / "quality_clustering.csv"
        with open(scores_path) as file:
            expected_scores = np.loadtxt(file, delimiter=",")
        np.testing.assert_allclose(
            dh.daily_scores.quality_clustering, expected_scores, atol=1e-3
        )
