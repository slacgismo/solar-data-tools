"""
Clear Sky Detection Module

This module contains functions to detect clear sky periods 
by applying a dynamic programming approach 
on the input data (cake) and estimated 98th percentile (Q98).
"""

import numpy as np

DEFAULT_CLEAR_SKY = {
    "lam": 2,
}

class ClearSkyDetection:
    def __init__(self, data_handler, **config):
        self.dh = data_handler
        self.cake = data_handler.cake  # assuming 'cake' is stored in the data_handler
        self.Q98 = data_handler.Q98    # assuming 'Q98' is stored in the data_handler
        self.D = self.cake.shape[0]    # Number of days
        self.T = self.cake.shape[1]    # Number of nodes in width
        if len(config) == 0:
            self.config = DEFAULT_CLEAR_SKY
        else:
            self.config = config
        self.lam = self.config["lam"]
        self.clearsky_cake = np.zeros_like(self.cake)
        self.run()

    def hinge_loss(self, y, q_tilde, type=0):
        return 1 if (y > q_tilde * 0.7 if type == 0 else y < q_tilde * 0.7) else 0
    
    def compute_hinge_losses(self, Y, Q98):
        nodelosses = np.zeros((2, self.T))
        for j in range(self.T):
            nodelosses[0, j] = self.hinge_loss(Y[j], Q98[j], type=0)
            nodelosses[1, j] = self.hinge_loss(Y[j], Q98[j], type=1)
        return nodelosses

    def find_optimal_path(self, nodelosses):
        cum_loss = nodelosses.copy()
        for t in range(1, self.T):
            cum_loss[0, t] += min(cum_loss[1, t-1] + self.lam, cum_loss[0, t-1])
            cum_loss[1, t] += min(cum_loss[0, t-1] + self.lam, cum_loss[1, t-1])
        path = np.zeros(self.T, dtype=int)
        path[-1] = np.argmin(cum_loss[:, -1])
        for t in range(self.T-2, -1, -1):
            prev_row = path[t+1]
            if cum_loss[prev_row, t] <= cum_loss[1-prev_row, t] + self.lam:
                path[t] = prev_row
            else:
                path[t] = 1 - prev_row
        return path

    def run(self):
        for i in range(self.D):
            nodelosses = self.compute_hinge_losses(self.cake[i], self.Q98[i])
            self.clearsky_cake[i] = self.find_optimal_path(nodelosses)
        return self.clearsky_cake

    def get_clearsky_cake(self):
        return self.clearsky_cake
