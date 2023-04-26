import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from solardatatools import DataHandler


class TestSystemProfiler(unittest.TestCase):
    def test_system_profiler(self):
        filepath = Path(__file__).parent.parent
        data_file_path = (
            filepath / "fixtures" / "system_profiler" / "data_handler_input.csv"
        )
        data = pd.read_csv(data_file_path, index_col=0, parse_dates=True)
        dh = DataHandler(data, datetime_col="Date-Time")
        dh.fix_dst()
        dh.run_pipeline(
            power_col="ac_power", fix_shifts=False, correct_tz=False, verbose=False
        )
        dh.setup_location_and_orientation_estimation(-5)

        estimate_latitude = dh.estimate_latitude()
        estimate_longitude = dh.estimate_longitude()

        estimate_orientation = dh.estimate_orientation()

        # Based on site documentation
        actual_latitude = 39.4856
        actual_longitude = -76.6636

        estimate_orientation_real_loc = dh.estimate_orientation(
            latitude=actual_latitude, longitude=actual_longitude
        )

        # Current algorithms output after changes in PR. Note, this
        # changes slightly each run, depending on the results of the
        # sunrise sunset evaluation, which uses holdout validation
        # to pick some hyperparameters.
        ref_latitude = 36.6  # +/- 0.3
        ref_longitude = -77.0  # +/- 0.03
        ref_tilt_real_loc = 22.45  # +/- 1e-4
        ref_az_real_loc = 0.28  # +/- 1e-6

        np.testing.assert_allclose(estimate_latitude, ref_latitude, atol=0.45)
        np.testing.assert_allclose(estimate_longitude, ref_longitude, atol=0.05)
        np.testing.assert_allclose(
            estimate_orientation_real_loc,
            (ref_tilt_real_loc, ref_az_real_loc),
            atol=0.05,
        )


if __name__ == "__main__":
    unittest.main()
