import unittest
import numpy as np
from solardatatools.plotting import plot_bundt_cake


class TestBundtPlot(unittest.TestCase):
    def test_bundt_plot(self):
        np.random.seed(42)
        test_data = np.random.rand(365, 100)
        _ = plot_bundt_cake(test_data)


if __name__ == "__main__":
    unittest.main()
