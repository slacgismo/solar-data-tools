# -*- coding: utf-8 -*-
""" Soiling Module

This module is for analyzing soiling trends in performance index (PI) data
and daily energy data.

"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler
import cvxpy as cvx
from gfosd import Problem
from gfosd.components import *

DEFAULT = {
    "degradation_term": True,
    "tau": 0.85,
    "iterations": 5,
    "w1": 2,
    "w2": 3e-2,
    "w3": 2e-1,
    "w4": 5e2,
}


class SoilingAnalysis:
    def __init__(self, data_handler):
        self.dh = data_handler
        e1 = data_handler.daily_signals.energy
        self.scaling = np.nanquantile(e1, 0.98)
        self.soiling_signal = None
        self.scaler_obj = None
        self.sd_problem = None
        self.soiling_decomposition = None

    def run(self, **config):
        self.preprocess_data()
        self.setup_sd_problem()
        self.sd_problem.decompose(max_iter=6000)
        out = self.undo_preprocess()
        self.decomposition = {
            "residual": out[0],
            "baseline": out[1] * self.scaling,
            "seasonal": out[2],
            "degradation": out[3],
            "soiling": out[4],
        }
        self.correction_factor = (
            self.decomposition["degradation"] * self.decomposition["soiling"]
        )
        self.corr_raw_matrix = self.dh.raw_data_matrix / self.correction_factor
        self.corr_filled_data_matrix = (
            self.dh.filled_data_matrix / self.correction_factor
        )

    def preprocess_data(self):
        e1 = self.dh.daily_signals.energy
        y0 = e1 / self.scaling
        # take log
        y1 = np.log(y0)
        # min-max scale t0 [0, 10]
        self.scaler_obj = MinMaxScaler(feature_range=(0, 10))
        y2 = np.ones_like(y1) * np.nan
        k = ~np.isnan(y1)
        y2[k] = self.scaler_obj.fit_transform(y1[k].reshape(-1, 1)).ravel()
        self.soiling_signal = y2

    def setup_sd_problem(self):
        # residual
        c1 = SumSquare(weight=1 / len(y))
        # bias term
        c2 = NoSlope()
        # seasonal baseline
        c3 = Aggregate(
            [SumSquare(weight=5e0, diff=2), Periodic(365), AverageEqual(0, period=365)]
        )
        # long-term linear degradation term
        c4 = Aggregate([NoCurvature(), FirstValEqual(0)])
        # soiling term
        c5 = Aggregate(
            [
                SumAbs(weight=1e-5),
                SumAbs(weight=5e-3, diff=2),
                Inequality(vmax=0),
                SumQuantile(weight=1e-5, diff=1, tau=0.9),
            ]
        )

        self.sd_problem = Problem(
            data=self.soiling_signal, components=[c1, c2, c3, c4, c5]
        )

    def undo_preprocess(self):
        X1 = np.zeros_like(self.sd_problem.decomposition)
        X1[1] = self.scaler_obj.inverse_transform(
            self.sd_problem.decomposition[1].reshape(-1, 1)
        ).ravel()
        X1[0] = self.sd_problem.decomposition[0] / self.scaler_obj.scale_
        X1[2:] = self.sd_problem.decomposition[2:] / self.scaler_obj.scale_
        X0 = np.exp(X1)
        return X0

    def plot_analysis(self, figsize=None):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(nrows=2, sharex=True, figsize=figsize)
        ax[0].plot(
            self.dh.day_index, self.dh.daily_signals.energy, label="signal", linewidth=1
        )
        d = self.decomposition
        ax[0].plot(
            self.dh.day_index,
            d["baseline"] * d["seasonal"] * d["degradation"] * d["soiling"],
            label="SD denoised model",
            linewidth=1,
        )
        ax[0].plot(
            self.dh.day_index,
            d["baseline"] * d["seasonal"] * d["degradation"],
            label="SD baseline",
            linewidth=1,
        )
        ax[0].legend()
        ax[0].set_title("signal and SD model")
        ax[1].plot(self.dh.day_index, d["soiling"], linewidth=1)
        ax[1].set_title("soiling component")
        return fig


def soiling_seperation(
    observed,
    index_set=None,
    degradation_term=DEFAULT["degradation_term"],
    tau=DEFAULT["tau"],
    w1=DEFAULT["w1"],
    w2=DEFAULT["w2"],
    w3=DEFAULT["w3"],
    w4=DEFAULT["w4"],
    iterations=DEFAULT["iterations"],
    solver=None,
    period=365,
    verbose=False,
):
    """
    Apply signal decomposition framework to Performance Index soiling estimation
    problem. The PI signal is a daily performance index, typically daily energy
    normalized by modeled or expected energy. PI signal assumed to contain
    components corresponding to

    (1) a soiling loss trend (sparse 1st-order differences)
    (2) a seasonal term (smooth, yearly periodic)
    (3) linear degradation
    (4) residual

    :param observed:
    :param index_set:
    :param degradation_term:
    :param tau:
    :param w1: PWL weight - soiling term
    :param w2: sparseness weight - soiling term
    :param w3: asymmetric slopes - soiling term
    :param w4: smoothness weight - seasonal term
    :param iterations:
    """
    if index_set is None:
        index_set = ~np.isnan(observed)
    else:
        index_set = np.logical_and(index_set, ~np.isnan(observed))
    # zero_set = np.zeros(len(observed) - 1, dtype=bool)
    eps = 0.01
    n = len(observed)
    s1 = cvx.Variable(n)  # soiling
    s2 = cvx.Variable(max(n, 367))  # seasonal
    s3 = cvx.Variable(n)  # degradation
    sr = cvx.Variable(n)  # residual
    # z = cvx.Variable(2)
    # T = len(observed)
    # M = np.c_[np.ones(T), np.arange(T)]
    w = cvx.Parameter(n - 2, nonneg=True)
    w.value = np.ones(len(observed) - 2)

    # cvx.norm(cvx.multiply(s3, weights), p=2) \

    length_scale_factor = 1  # len(observed) / 3653
    cost = (
        cvx.sum(tau * cvx.pos(sr) + (1 - tau) * cvx.neg(sr))
        + w4 * cvx.norm(cvx.diff(s2[:n], k=2), p=2)
        + length_scale_factor
        * (
            w1 * cvx.norm(cvx.multiply(w, cvx.diff(s1, k=2)), p=1)
            + w2 * cvx.sum(-s1)
            + w3
            * (
                cvx.sum(
                    0.9 * cvx.pos(cvx.diff(s1, k=1))
                    + (1 - 0.1) * cvx.neg(cvx.diff(s1, k=1))
                )
            )
        )
    )
    objective = cvx.Minimize(cost)
    constraints = [
        observed[index_set] == (s1 + s2[:n] + s3 + sr)[index_set],
        s2[period:] - s2[:-period] == 0,
        # cvx.sum(s2[:period]) == 0,
        s1 <= 0,
    ]
    if degradation_term:
        constraints.extend([cvx.diff(s3, k=2) == 0, s3[0] == 0])
    else:
        constraints.append(s3 == 0)
    # if np.sum(zero_set) > 0:
    #     constraints.append(cvx.diff(s1, k=1)[zero_set] == 0)
    if n < 0.75 * 365:
        constraints.append(s2 == 0)
    problem = cvx.Problem(objective, constraints)
    for i in range(iterations):
        problem.solve(solver=solver, verbose=verbose)
        w.value = 1 / (
            eps + 1e2 * np.abs(cvx.diff(s1, k=2).value)
        )  # Reweight the L1 penalty
        # zero_set = np.abs(cvx.diff(s1, k=1).value) <= 5e-5     # Make nearly flat regions exactly flat (sparse 1st diff)
    out = {
        "soiling": s1.value,
        "seasonal": s2.value[:n],
        "trend": s3.value,
        "residual": sr.value,
        "denoised": s1.value + s2.value[:n] + s3.value,
    }
    return out


def soiling_seperation_old(
    observed,
    index_set=None,
    degradation_term=False,
    period=365,
    tau=0.85,
    w1=2,
    w2=1e-2,
    w3=100,
    iterations=5,
    soiling_max=1.0,
    solver="MOSEK",
):
    """
    Apply signal decomposition framework to Performance Index soiling estimation
    problem. The PI signal is a daily performance index, typically daily energy
    normalized by modeled or expected energy. PI signal assumed to contain
    components corresponding to

    (1) a soiling loss trend (sparse 1st-order differences)
    (2) a seasonal term (smooth, yearly periodic)
    (3) linear degradation
    (4) residual

    :param observed:
    :param index_set:
    :param degradation_term:
    :param tau:
    :param w1: PWL weight - soiling term
    :param w2: sparseness weight - soiling term
    :param w3: smoothness weight - seasonal term
    :param iterations:
    """
    if index_set is None:
        index_set = ~np.isnan(observed)
    else:
        index_set = np.logical_and(index_set, ~np.isnan(observed))
    # zero_set = np.zeros(len(observed) - 1, dtype=bool)
    eps = 0.01
    n = len(observed)
    s1 = cvx.Variable(n)  # soiling
    s2 = cvx.Variable(max(n, 367))  # seasonal
    s3 = cvx.Variable(n)  # degradation
    sr = cvx.Variable(n)  # residual
    # z = cvx.Variable(2)
    # T = len(observed)
    # M = np.c_[np.ones(T), np.arange(T)]
    w = cvx.Parameter(n - 2, nonneg=True)
    w.value = np.ones(len(observed) - 2)

    # cvx.norm(cvx.multiply(s3, weights), p=2) \

    cost = (
        cvx.sum(tau * cvx.pos(sr) + (1 - tau) * cvx.neg(sr))
        + w3 * cvx.norm(cvx.diff(s2[:n], k=2), p=2)
        + w1 * cvx.norm(cvx.multiply(w, cvx.diff(s1, k=2)), p=1)
        + w2 * cvx.sum(soiling_max - s1)
    )
    objective = cvx.Minimize(cost)
    constraints = [
        observed[index_set] == (s1 + s2[:n] + s3 + sr)[index_set],
        s2[period:] - s2[:-period] == 0,
        cvx.sum(s2[:period]) == 0,
        s1 <= soiling_max,
    ]
    if degradation_term:
        constraints.extend([cvx.diff(s3, k=2) == 0, s3[0] == 0])
    else:
        constraints.append(s3 == 0)
    # if np.sum(zero_set) > 0:
    #     constraints.append(cvx.diff(s1, k=1)[zero_set] == 0)
    if n < 0.75 * 365:
        constraints.append(s2 == 0)
    problem = cvx.Problem(objective, constraints)
    for i in range(iterations):
        problem.solve(solver=solver)
        w.value = 1 / (
            eps + 1e2 * np.abs(cvx.diff(s1, k=2).value)
        )  # Reweight the L1 penalty
        # zero_set = np.abs(cvx.diff(s1, k=1).value) <= 5e-5     # Make nearly flat regions exactly flat (sparse 1st diff)
    return s1.value, s2.value[:n], s3.value, sr.value
