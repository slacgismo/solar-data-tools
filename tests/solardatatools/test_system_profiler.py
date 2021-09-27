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

        dh = DataHandler(data)
        dh.fix_dst()
        dh.run_pipeline(power_col="ac_power", fix_shifts=False, correct_tz=False)
        dh.setup_location_and_orientation_estimation(-5)

        estimate_latitude = dh.estimate_latitude()
        estimate_longitude = dh.estimate_longitude()

        estimate_orientation = dh.estimate_orientation()

        actual_latitude = 39.4856
        actual_longitude = -76.6636

        actual_orientation = dh.estimate_orientation(
            latitude=actual_latitude, longitude=actual_longitude
        )

        np.testing.assert_almost_equal(estimate_latitude, actual_latitude, decimal=0.5)
        np.testing.assert_almost_equal(
            estimate_longitude, actual_longitude, decimal=0.5
        )
        np.testing.assert_array_almost_equal(
            estimate_orientation, actual_orientation, decimal=0.1
        )


if __name__ == "__main__":
    unittest.main()
