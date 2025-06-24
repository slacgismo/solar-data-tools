"""This module contains tests for two algorithm modules:

        1) quantile estimation
        2) time dilation

TESTS
-----
-
"""

import unittest
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from solardatatools import DataHandler
import contextlib
import sys
from io import StringIO


class TestQuantileDilation(unittest.TestCase):
    def setUp(self):
        self.filepath = (
            Path(__file__).parent.parent / "fixtures" / "quantile_estimation"
        )

    def test_quantile_and_dilation(self):
        # load input data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            input = pd.read_csv(
                self.filepath / "test_power.csv", index_col=0, parse_dates=[0]
            )
        # load expected outputs
        signal_dil = np.loadtxt(self.filepath / "signal_dil.txt")
        quant_dil_df = pd.read_csv(self.filepath / "quant_dil_df.csv")
        quant_ori_df = pd.read_csv(self.filepath / "quant_ori_df.csv")
        # run code
        dh = DataHandler(input)
        dh.run_pipeline(power_col="ac_power_inv_30339", verbose=False)
        with silenced():
            dh.estimate_quantiles(
                nvals_dil=21,
                quantile_levels=[0.2, 0.5, 0.8],
                num_harmonics=[8, 3],
                regularization=0.5,
                verbose=False,
            )
        # check dilation
        dil_entries_close = np.isclose(
            dh.quantile_object.dilation_object.signal_dil,
            signal_dil,
            equal_nan=True,
            atol=1e-1,
        )
        # total of 7687 entries, there is some variability in this operation, and
        # up to 25 entries have been observed to not match on some runs
        self.assertLessEqual(np.sum(~dil_entries_close), 50)
        # check quantiles
        ql = dh.quantile_object.quantile_levels
        for q in ql:
            np.testing.assert_array_almost_equal(
                actual=dh.quantile_object.quantiles_dilated[q],
                desired=quant_dil_df[str(q)],
                decimal=1,
            )
            q_ori_close = np.isclose(
                dh.quantile_object.quantiles_original[q],
                quant_ori_df[str(q)],
                atol=1e-1,
            )
            # total 105408 entries
            self.assertLessEqual(np.sum(~q_ori_close), 750)


@contextlib.contextmanager
def silenced(no_stdout=True, no_stderr=True):
    """
    Suppresses output to stdout and/or stderr.
    Always resets stdout and stderr, even on an exception.
    Usage:
        with silenced(): print("This doesn't print")
    Via https://gist.github.com/dmyersturnbull/c0717406eb46158401a9
    Modified from post by Alex Martelli in https://stackoverflow.com/questions/2828953/silence-the-stdout-of-a-function-in-python-without-trashing-sys-stdout-and-resto/2829036#2829036
    which is licensed under CC BY-SA 3.0 https://creativecommons.org/licenses/by-sa/3.0/
    """
    if no_stdout:
        save_stdout = sys.stdout
        sys.stdout = StringIO()
    if no_stderr:
        save_stderr = sys.stderr
        sys.stderr = StringIO()
    yield
    if no_stdout:
        sys.stdout = save_stdout
    if no_stderr:
        sys.stderr = save_stderr


if __name__ == "__main__":
    unittest.main()
