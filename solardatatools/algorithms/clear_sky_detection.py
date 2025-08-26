"""
Clear Sky Detection Module
Author: Mehmet Giray Ogut
Date: June 11, 2025
This module contains functions to detect clear sky periods
by applying a dynamic programming approach
on the input data (cake) and estimated 98th percentile (Q98).
"""

import numpy as np


class ClearSkyDetection:
    def __init__(
        self,
        sig,
        quantile_estimate,
        threshold_low=0.75,
        threshold_high=1.2,
        stickiness_low=4,
        stickiness_high=0.1,
    ):
        """
        A class for detecting clear sky periods in measured PV power data. This class requires the user
        to provide a signal to be labeled, a time-series quantile estimate of that signal, and a number
        of parameters that control the dynamic programming algorithm.

        The basic model is that the ratio of the measured signal to the quantile estimate is between a band
        defined by threshold_low and threshold_high when conditions are "clear sky", and the ratio
        is outside that band when conditions are not "clear sky". The "stickiness" parameters define the cost
        to transition out of a cloudy state (stickiness_low) or out of a clear state (stickiness_high). We
        set stickiness_low > stickiness_high so that it is easier to transition from clear to cloudy than from
        cloudy to clear.

        :param sig: the measured PV power signal to be labeled
        :type sig: 1D numpy.array object
        :param quantile_estimate: the estimated "clear sky" quantile, typically the 0.9 level
        :type sig: 1D numpy.array object of same length as sig
        :param theshold_low: the lower boundry of the clear sky ratio band
        :type theshold_low: float
        :param theshold_high: the upper boundry of the clear sky ratio band
        :type theshold_high: float
        :param stickiness_low: the cost to transition out of the cloudy state (large)
        :type stickiness_low: float
        :param stickiness_high: the cost to transition out of the clear sky state (small)
        :type stickiness_high: float
        """
        assert len(sig) == len(quantile_estimate), (
            "signal and quantile estimate must have the same length!"
        )
        self.stickiness_low = stickiness_low
        self.stickiness_high = stickiness_high
        self.sig = np.atleast_1d(sig)
        self.T = self.sig.shape[0]
        self.Q98 = np.atleast_1d(quantile_estimate)
        self.threshold_low = threshold_low
        self.threshold_high = threshold_high
        self.clearsky_sig = np.zeros_like(self.sig, dtype=int)
        self.run()

    def hinge0(self, val, q98):
        return (
            0.0
            if (val <= q98 * self.threshold_low or val >= q98 * self.threshold_high)
            else 1.0
        )

    def hinge1(self, val, q98):
        return (
            0.0
            if (val >= q98 * self.threshold_low and val <= q98 * self.threshold_high)
            else 1.0
        )

    def compute_hinge_losses(self, values, q98_row):
        losses = np.zeros((2, self.T))
        for j in range(self.T):
            val = values[j]
            q98 = q98_row[j]
            losses[0, j] = self.hinge0(val, q98)
            losses[1, j] = self.hinge1(val, q98)
        return losses

    def find_optimal_path(self, L):
        cum_loss = L.copy()
        for t in range(1, self.T):
            cum_loss[0, t] += min(
                cum_loss[1, t - 1] + self.stickiness_high, cum_loss[0, t - 1]
            )
            cum_loss[1, t] += min(
                cum_loss[0, t - 1] + self.stickiness_low, cum_loss[1, t - 1]
            )
        Z = np.argmin(cum_loss, axis=0)
        # Z = np.zeros(self.T, dtype=int)
        # Z[-1] = np.argmin(cum_loss[:, -1])
        # for t in range(self.T - 2, -1, -1):
        #     prev = Z[t + 1]
        #     if prev == 0:
        #         Z[t] = prev if (cum_loss[prev, t]
        #                         <= cum_loss[1 - prev, t] + self.stickiness_high) else 1 - prev
        #     if prev == 1:
        #         Z[t] = prev if (cum_loss[prev, t]
        #                         <= cum_loss[1 - prev, t] + self.stickiness_low) else 1 - prev
        return Z

    def run(self):
        losses = self.compute_hinge_losses(self.sig, self.Q98)
        self.clearsky_sig = self.find_optimal_path(losses)
        return self.clearsky_sig

    def get_clearsky_sig(self):
        return self.clearsky_sig
