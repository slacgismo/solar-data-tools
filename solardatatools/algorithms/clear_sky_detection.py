"""
Clear Sky Detection Module

This module contains functions to detect clear sky periods 
by applying a dynamic programming approach 
on the input data (cake) and estimated 98th percentile (Q98).
"""

import numpy as np
from solardatatools.algorithms.dilation import Dilation  # Import the Dilation class

DEFAULT_CLEAR_SKY = {
    "lam": 2,
}


class ClearSkyDetection:
    def __init__(self, data_handler, cake=None, **config):
        self.dh = data_handler
        self.lam = config.pop("lam", DEFAULT_CLEAR_SKY["lam"])

        if cake is None:
            raise Exception("Bundt cake is not provided.")
        self.cake = cake

        self.Q98 = data_handler.Q98
        self.D = self.cake.shape[0]
        self.T = self.cake.shape[1]
        self.clearsky_cake = np.zeros_like(self.cake)
        self.run()

    def hinge0(self, val, q98):
        return 0.0 if val <= q98 * 0.7 else 1.0

    def hinge1(self, val, q98):
        return 0.0 if val >= q98 * 0.7 else 1.0

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
            for i in range(2):
                cum_loss[i, t] += min(cum_loss[1 - i, t - 1] +
                                      self.lam, cum_loss[i, t - 1])
        Z = np.zeros(self.T, dtype=int)
        Z[-1] = np.argmin(cum_loss[:, -1])
        for t in range(self.T - 2, -1, -1):
            prev = Z[t + 1]
            Z[t] = prev if cum_loss[prev, t] <= cum_loss[1 -
                                                         prev, t] + self.lam else 1 - prev
        return Z

    def run(self):
        for i in range(self.D):
            values = self.cake[i]
            q98_row = self.Q98[i]
            losses = self.compute_hinge_losses(values, q98_row)
            self.clearsky_cake[i] = self.find_optimal_path(losses)
        return self.clearsky_cake

    def get_clearsky_cake(self):
        return self.clearsky_cake
