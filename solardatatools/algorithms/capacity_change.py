""" Capacity Change Algorithm Module

This module the algorithm for detecting capacity changes in an unlabeled PV
power production data sets. The algorithm works as follows:

    - Run daily quantile statistic on cleaned data
    - Fit a signal demixing model, assuming a seasonal component and a piecewise
      constant component
    - Polish the L1 heuristic used to estimate piecewise constant component
      using iterative reweighting
    - Assign daily cluster labels using DBSCAN algorithm



"""

import numpy as np
from solardatatools.osd_signal_decompositions import tl1_l1d1_l2d2p365
from sklearn.cluster import DBSCAN


class CapacityChange:
    def __init__(self):
        self.metric = None
        self.s1 = None
        self.s2 = None
        self.labels = None

    def run(
        self,
        data,
        filter=None,
        quantile=1.00,
        c1=15,
        c2=6561,
        c3=300,
        tau=0.5,
        reweight_eps=0.5,
        reweight_niter=5,
        dbscan_eps=0.02,
        dbscan_min_samples="auto",
        solver=None
    ):
        if filter is None:
            filter = np.ones(data.shape[1], dtype=bool)
        if np.sum(filter) > 0:
            metric = np.nanquantile(data, q=quantile, axis=0)
            # metric = np.sum(data, axis=0)
            metric /= np.max(metric)

            def max_min_scale(signal):
                maximum = np.nanquantile(signal, .95)
                minimum = np.nanquantile(signal, .05)
                return (signal - minimum) / (maximum - minimum), minimum, maximum

            def rescale_signal(signal, minimum, maximum):
                return (signal * (maximum - minimum)) + minimum

            scaled_metric, min_metric, max_metric = max_min_scale(metric)

            # w = np.ones(len(metric) - 1)
            # eps = reweight_eps

            # for i in range(reweight_niter):
            s1, s2 = tl1_l1d1_l2d2p365(
                metric,
                use_ixs=filter,
                tau=tau,
                w1=c1,
                w2=c2,
                w3=c3,
                # tv_weights=w,
                solver=solver,
            )
                # w = 1 / (eps + np.abs(np.diff(s1, n=1)))
        else:
            # print('No valid values! Please check your data and filter.')
            return

        # s1 = rescale_signal(
        #     s1,
        #     min_metric,
        #     max_metric
        # )
        #
        # s2 = rescale_signal(
        #     s2,
        #     min_metric,
        #     max_metric
        # )

        if dbscan_eps is None or dbscan_min_samples is None:
            self.metric = metric
            self.s1 = s1
            self.s2 = s2
            self.labels = None
        else:
            if dbscan_min_samples == "auto":
                dbscan_min_samples = int(max(0.1 * len(s1), 3))
            db = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit(
                s1[:, np.newaxis]
            )
            capacity_assignments = db.labels_
            self.metric = metric
            self.s1 = s1
            self.s2 = s2
            self.labels = capacity_assignments
