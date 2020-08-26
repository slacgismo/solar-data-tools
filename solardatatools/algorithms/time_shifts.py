''' Time Shift Algorithm Module

This module contains the algorithm for detecting time shifts in an unlabeled PV
power production data sets. These occur because of the local clock on the data
logging system being changed or by incorrect handling of daylight savings.
The algorithm works as follows:

    - Estimate solar noon on each day from the data
    - Fit a signal demixing model, assuming a seasonal component and a piecewise
      constant component
    - Polish the L1 heuristic used to estimate piecewise constant component
      using iterative reweighting
    - Use piecewise constance component to detect shift points in time and
      correction amounts

'''

import numpy as np
from scipy.stats import mode
from solardatatools.solar_noon import energy_com, avg_sunrise_sunset
from solardatatools.utilities import total_variation_plus_seasonal_filter

class TimeShift():
    def __init__(self):
        self.metric = None
        self.s1 = None
        self.s2 = None
        self.index_set = None
        self.corrected_data = None

    def run(self, data, use_ixs=None, c1=5., c2=200.,
            solar_noon_estimator='com', threshold=0.1):
        if solar_noon_estimator == 'com':
            metric = energy_com(data)
        elif solar_noon_estimator == 'srss':
            metric = avg_sunrise_sunset(data, threshold=threshold)
        if use_ixs is None:
            use_ixs = ~np.isnan(metric)
        # Iterative reweighted L1 heuristic
        w = np.ones(len(metric) - 1)
        eps = 0.1
        for i in range(5):
            s1, s2 = total_variation_plus_seasonal_filter(
                metric, c1=c1, c2=c2,
                tv_weights=w,
                use_ixs=use_ixs
            )
            w = 1 / (eps + np.abs(np.diff(s1, n=1)))
        # Apply corrections
        roll_by_index = np.round(
            (mode(np.round(s1, 3)).mode[0] - s1) * data.shape[0] / 24, 0)
        Dout = np.copy(data)
        for roll in np.unique(roll_by_index):
            if roll != 0:
                ixs = roll_by_index == roll
                Dout[:, ixs] = np.roll(data, int(roll), axis=0)[:, ixs]
        # find indices of transition points
        index_set = np.arange(len(s1) - 1)[np.round(np.diff(s1, n=1), 0) != 0]
        # save results
        self.metric = metric
        self.s1 = s1
        self.s2 = s2
        self.index_set = index_set
        self.corrected_data = Dout