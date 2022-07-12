""" Shade Module

This module is for analyzing shade losses in unlabeled power data

"""

import numpy as np
import pandas as pd
import cvxpy as cvx
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from solardatatools.polar_transform import PolarTransform

my_round = lambda x, c: c * np.round(x / c, 0)

FP = Path(__file__).parent.parent
BASIS = np.loadtxt(FP / "fixtures" / "clear_PCA.txt")
Q = np.loadtxt(FP / "fixtures" / "Q.txt")
RANK = 6
Qr = Q[:, :RANK]
LAMBD = np.loadtxt(FP / "fixtures" / "eigvals.txt")
LAMBDr = LAMBD[:RANK]
MU = np.loadtxt(FP / "fixtures" / "mu.txt")
DATA_CORPUS = pd.read_csv(
    FP / "fixtures" / "transformed_data_corpus.zip", index_col=np.arange(4)
)


class ShadeAnalysis:
    def __init__(self, data_handler, matrix=None):
        self.dh = data_handler
        if matrix is None:
            self.data = self.dh.filled_data_matrix
        else:
            self.data = matrix
        self.data_normalized = None
        self.data_transformed = None
        self.osd_problem = None
        self.residual_component = None
        self.clear_sky_component = None
        self.shade_component = None
        self.residual_estimate = None
        self.clear_sky_estimate = None
        self.shade_estimate = None
        self.power_estimate = None
        self.year_analysis_df = None
        self.scale_factor = 1 / self.dh.capacity_estimate
        self.unrolled_shade = None
        self._pt1 = None

    @property
    def has_run(self):
        state = np.alltrue(
            [
                self.data_normalized is not None,
                self.data_transformed is not None,
                self.osd_problem is not None,
            ]
        )
        return state

    def run(
        self, power=8, solver="MOSEK", verbose=False, mu=None, lambd=None, q_mat=None
    ):
        if not self.has_run:
            dn, dt = self.transform_data(power)
            self.data_normalized = dn
            self.data_transformed = dt
            self.osd_problem = self.make_osd_problem(mu=mu, lambd=lambd, q_mat=q_mat)
            self.osd_problem.solve(solver=solver, verbose=verbose)
        variable_dict = {v.name(): v for v in self.osd_problem.variables()}
        self.clear_sky_component = variable_dict["clear-sky"].value / self.scale_factor
        self.shade_component = variable_dict["shade"].value / self.scale_factor
        self.residual_component = variable_dict["residual"].value / self.scale_factor
        # create estimates in shape of original data
        self.clear_sky_estimate = undo_batch_process(
            self._unroll_estimate(self.clear_sky_component),
            self.dh.boolean_masks.daytime,
        )
        self.shade_estimate = undo_batch_process(
            self._unroll_estimate(self.shade_component), self.dh.boolean_masks.daytime
        )
        self.power_estimate = self.clear_sky_estimate - self.shade_estimate
        self.residual_estimate = undo_batch_process(
            self._unroll_estimate(self.residual_component),
            self.dh.boolean_masks.daytime,
        )

    def analyze_yearly_energy(self):
        if not self.has_run:
            self.run()

        # get scale factor for converting column sum to energy
        N = self.dh.data_sampling
        scale = N / 60
        # construct function to map from declination angle to total daily
        # integral of component
        f_loss = interp1d(
            self.data_transformed.index.values,
            np.sum(self.shade_component, axis=1) * scale,
            bounds_error=False,
            fill_value="extrapolate",
        )
        f_clear = interp1d(
            self.data_transformed.index.values,
            np.sum(self.clear_sky_component, axis=1) * scale,
            bounds_error=False,
            fill_value="extrapolate",
        )

        avg_energy = pd.DataFrame(
            data=np.sum(self.data, axis=0), index=self.dh.day_index
        )
        avg_energy["doy"] = avg_energy.index.day_of_year
        avg_energy = avg_energy.loc[self.dh.daily_flags.clear]
        avg_energy = avg_energy.groupby("doy").mean() * scale
        avg_energy.columns = ["empirical"]
        sl = f_loss(delta_cooper(np.arange(365) + 1))
        cs = f_clear(delta_cooper(np.arange(365) + 1))
        self.year_analysis_df = pd.DataFrame(
            data={"shade loss": sl, "clear sky": cs, "SD model": cs - sl},
            index=np.arange(365) + 1,
        )
        self.year_analysis_df = self.year_analysis_df.join(avg_energy)

    def plot_yearly_energy_analysis(self, figsize=None):
        fig = plt.figure(figsize=figsize)
        plt.plot(self.year_analysis_df["shade loss"], label="shade loss")
        plt.plot(self.year_analysis_df["clear sky"], label="predicted no shade")
        plt.plot(self.year_analysis_df["SD model"], label="modeled")
        plt.plot(
            self.year_analysis_df.index,
            self.year_analysis_df["empirical"],
            label="empirical",
            linewidth=0.75,
        )
        plt.xlabel("day of year")
        plt.ylabel("energy [kWh]")
        plt.title("Seasonal energy analysis")
        plt.legend()
        return fig

    def plot_transformed_data(
        self,
        yticks=True,
        figsize=(10, 4),
        cmap="plasma",
        interpolation="none",
        aspect="auto",
    ):
        plt.figure(figsize=figsize)
        ax = plt.gca()
        foo = ax.imshow(
            self.data_transformed.values,
            cmap=cmap,
            interpolation=interpolation,
            aspect=aspect,
        )
        plt.colorbar(foo, ax=ax, label="norm. power")
        # sns.heatmap(self.data_transformed, cmap="plasma")
        plt.xticks(np.linspace(0, 256, 11), np.round(np.linspace(0, 1, 11), 1))
        plt.xlabel("fraction of daylight hours")
        if not yticks:
            plt.yticks([])
        else:
            num_y = self.data_transformed.values.shape[0]
            max_d = np.max(self.data_transformed.index)
            plt.yticks(
                np.linspace(0, num_y, 9), np.round(np.linspace(-max_d, max_d, 9), 1)
            )
        return plt.gcf()

    def _unroll_normalized_shade_loss(self):
        M = self.data_transformed.shape[1]
        unrolled = np.zeros((M, self.dh.num_days))
        metric = np.zeros_like(self.shade_component)
        # cond = ~np.isclose(self.clear_sky_component, 0)
        cond = self.clear_sky_component >= 0.02 * np.nanquantile(
            self.clear_sky_component, 0.98
        )
        metric[cond] = self.shade_component[cond] / self.clear_sky_component[cond]
        for ix, d in enumerate(
            my_round(delta_cooper(self.dh.day_index.dayofyear.values), 1)
        ):
            slct = self.data_transformed.index == d
            unrolled[:, ix] = metric[slct, :]
        self.unrolled_shade = unrolled

    def _unroll_estimate(self, estimate):
        """
        Creates the "normalized" view of the data, with no nighttime and
        M data points per day. The energy content of the columns are correct.
        """
        M = self.data_transformed.shape[1]
        unrolled = np.zeros((M, self.dh.num_days))
        for ix, d in enumerate(
            my_round(delta_cooper(self.dh.day_index.dayofyear.values), 1)
        ):
            # print('unrolling, delta =', d)
            slct = self.data_transformed.index == d
            unrolled[:, ix] = estimate[slct, :]
        return unrolled

    def plot_annotated_polar(
        self,
        lat,
        lon,
        tz_offset,
        elevation_round=1,
        azimuth_round=2,
        t=0.25,
        figsize=(10, 6),
    ):
        # print('starting')
        if self.unrolled_shade is None:
            self._unroll_normalized_shade_loss()
        unrolled = self.unrolled_shade
        # print('1')
        bool_mask = undo_batch_process(unrolled, self.dh.boolean_masks.daytime) >= t
        # print('2')
        self.dh.augment_data_frame(bool_mask, "shade_detect")
        # print('3')
        fig = self.dh.plot_polar_transform(
            lat=lat,
            lon=lon,
            tz_offset=tz_offset,
            elevation_round=elevation_round,
            azimuth_round=azimuth_round,
            alpha=1,
        )
        ax = fig.axes[0]
        # print('4')
        pt2 = PolarTransform(
            self.dh.data_frame["shade_detect"].astype(float),
            lat,
            lon,
            tz_offset=tz_offset,
            boolean_selection=self.dh.data_frame["clear-day"],
            normalize_data=False,
        )

        # print('5')
        pt2.transform(
            agg_func=np.nanmean,
            elevation_round=elevation_round,
            azimuth_round=azimuth_round,
        )
        # print('6')
        pt2.transformed_data = pt2.transformed_data.round()
        pt2.transformed_data[np.isclose(pt2.transformed_data, 0)] = np.nan
        # print('7')
        fig = pt2.plot_transformation(ax=ax, cmap="Set1", alpha=0.40, cbar=False)
        plt.scatter(0, 0, color="red", alpha=0.55, label="detected shade", marker="s")
        plt.legend()
        return fig

    def plot_annotated_heatmap(self, t=0.25, figsize=(12, 6)):
        if self.unrolled_shade is None:
            self._unroll_normalized_shade_loss()
        unrolled = self.unrolled_shade
        fig = self.dh.plot_heatmap(matrix="filled", figsize=figsize)
        annotate = undo_batch_process(unrolled, self.dh.boolean_masks.daytime) >= t
        annotate = np.asarray(annotate, dtype=float)
        annotate[annotate == 0] = np.nan
        with sns.axes_style("white"):
            plt.imshow(annotate, aspect="auto", cmap="Set1", alpha=0.5)
            plt.scatter(0, 0, color="red", alpha=0.45, label="detected shade")
            plt.legend()
        plt.title("Measured power with shaded periods marked")
        return fig

    def plot_component(self, component, figsize=(10, 4), ax=None, cmap="plasma"):
        if component == "clear":
            data = self.clear_sky_component
            title = "clear sky component"
        elif component == "shade":
            data = -self.shade_component
            title = "shade loss component"
        elif component == "residual":
            data = self.residual_component
            title = "residual component"
        else:
            m = "component arg must be one of ['clear', 'shade', 'residual']"
            print(m)
            return
        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = plt.gca()
        else:
            fig = None
        with sns.axes_style("white"):
            if not component == "residual":
                foo = ax.imshow(data, aspect="auto", cmap=cmap)
            else:
                val = max(np.max(data), np.max(-1 * data))
                foo = ax.imshow(data, aspect="auto", cmap=cmap, vmin=-val, vmax=val)
            plt.colorbar(foo, ax=ax)
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("fraction daylight hours")
            ax.set_ylabel("azimuth at sunrise")
        return fig

    def transform_data(self, power=8):
        normalized = batch_process(
            self.data,
            self.dh.boolean_masks.daytime,
            power=power,
            scale=self.scale_factor,
        )
        agg_by_azimuth = pd.DataFrame(
            data=normalized.T,
            index=np.arange(normalized.shape[1]),
            columns=np.linspace(0, 1, 2**power),
        )
        agg_by_azimuth["delta"] = my_round(
            delta_cooper(self.dh.day_index.dayofyear.values), 1
        )
        # select only clear days
        agg_by_azimuth = agg_by_azimuth.iloc[self.dh.daily_flags.clear]
        agg_by_azimuth = agg_by_azimuth.groupby("delta").mean()
        agg_by_azimuth = agg_by_azimuth.iloc[::-1]
        agg_by_azimuth.columns = np.round(
            np.asarray(agg_by_azimuth.columns, dtype=float), 4
        )
        new_index = np.arange(
            np.min(agg_by_azimuth.index), np.max(agg_by_azimuth.index) + 1
        )[::-1]
        agg_by_azimuth = agg_by_azimuth.reindex(new_index)
        return normalized, agg_by_azimuth

    def make_osd_problem(self, mu=None, lambd=None, q_mat=None):
        if mu is None:
            mu = MU
        if lambd is None:
            lambd = LAMBDr
        if q_mat is None:
            q_mat = Qr
        rank = q_mat.shape[1]
        y = self.data_transformed.values
        use_set = ~np.isnan(y)
        x1 = cvx.Variable(y.shape, name="residual")
        x2 = cvx.Variable(y.shape, name="clear-sky")
        x3 = cvx.Variable(y.shape, name="shade")
        t = 0.95

        # z2 = cvx.Variable((y.shape[0], BASIS.shape[0]), name='clear-sky-basis')
        z2 = cvx.Variable((rank, y.shape[0]), name="clear-sky-basis")
        M = cvx.Parameter(
            (y.shape[1], rank),
            value=np.block(
                [
                    [np.diag(np.sqrt(np.divide(1.0, lambd)))],
                    [np.zeros((y.shape[1] - rank, rank))],
                ]
            ),
        )

        # phi1 = cvx.sum_squares(x1)
        phi1 = cvx.sum(0.5 * cvx.abs(x1) + (t - 0.5) * x1)
        #### Original formulation based on smoothness
        # phi2 = cvx.sum_squares(cvx.diff(x2, k=2, axis=1))
        #### Simple basis constraint formulation---incorrect because it
        # doesn't include the power spectrum
        # phi2 = cvx.sum_squares(z2 @ BASIS - x2)
        #### Truncated covariance matrix class
        phi2 = 0.5 * cvx.sum_squares(M @ z2)
        #### add smoothness along vertical axis
        phi2 += 1e3 * cvx.sum_squares(cvx.diff(x2, k=2, axis=0))

        phi3 = cvx.sum_squares(cvx.diff(x3, k=2, axis=0)) + cvx.sum_squares(
            cvx.diff(x3, k=2, axis=1)
        )
        phi4 = cvx.sum(x3)

        constraints = [
            y[use_set] == (x1 + x2 - x3)[use_set],
            x2 >= 0,
            cvx.diff(x2, k=2, axis=0) <= 0,
            x2[:, 0] == 0,
            x2[:, -1] == 0,
            # cvx.diff(x2, k=4, axis=1) <= 0,
            x2 - np.tile(mu, (y.shape[0], 1)) == (q_mat @ z2).T,
            x3 >= 0,
        ]

        objective = cvx.Minimize(20 * phi1 + 1e0 * phi2 + 5e2 * phi3 + 3e-1 * phi4)
        problem = cvx.Problem(objective, constraints)
        return problem


def batch_process(data, mask, power=8, scale=None):
    """Process an entire PV power matrix at once
    :return:
    """
    if scale is None:
        scale = 1
    N = 2**power
    output = np.zeros((N, data.shape[1]))
    xs_new = np.linspace(0, 1, N)
    for col_ix in range(data.shape[1]):
        y = data[:, col_ix] * scale
        energy = np.sum(y)
        msk = mask[:, col_ix]
        xs = np.linspace(0, 1, int(np.sum(msk)))
        interp_f = interp1d(xs, y[msk])
        resampled_signal = interp_f(xs_new)
        if np.sum(resampled_signal) > 0:
            output[:, col_ix] = resampled_signal * energy / np.sum(resampled_signal)
        else:
            output[:, col_ix] = 0
    return output


def undo_batch_process(data, mask, scale=None):
    if scale is None:
        scale = 1
    output = np.zeros_like(mask, dtype=float)
    xs_old = np.linspace(0, 1, data.shape[0])
    for col_ix in range(data.shape[1]):
        # the number of non-zero data points on that day
        n_pts = np.sum(mask[:, col_ix])
        # a linear space of n_pts values between 0 and 1
        xs_new = np.linspace(0, 1, n_pts)
        # the energy content of each column before the transformation
        energy = np.sum(data[:, col_ix])
        # the mapping from the old index to the data
        interp_f = interp1d(xs_old, data[:, col_ix] / scale)
        # transform the signal to the new length
        resampled_signal = interp_f(xs_new)
        # correct the energy contet
        resampled_signal = resampled_signal * energy / np.sum(resampled_signal)
        # insert the data into the daytime index values
        output[mask[:, col_ix], col_ix] = resampled_signal
    return output


def delta_cooper(day_of_year):
    """ "
    Declination delta is estimated from equation (1.6.1a) in:
    Duffie, John A., and William A. Beckman. Solar engineering of thermal
    processes. New York: Wiley, 1991.
    """
    delta_1 = 23.45 * np.sin(np.deg2rad(360 * (284 + day_of_year) / 365))
    return delta_1


def make_class_parameters(az_max=None, az_min=None, tl_max=None, tl_min=None, rank=6):
    query = []
    if az_max is not None:
        query.append("az <= {}".format(az_max))
    if az_min is not None:
        query.append("az >= {}".format(az_min))
    if tl_max is not None:
        query.append("tl <= {}".format(tl_max))
    if tl_min is not None:
        query.append("tl >= {}".format(tl_min))
    if len(query) == 0:
        data = DATA_CORPUS.values
    else:
        query = " & ".join(query)
        data = DATA_CORPUS.query(query).values
    mu = np.average(data, axis=0)
    data_tilde = data - mu
    cov = data_tilde.T @ data_tilde / data_tilde.shape[0]
    evals, evecs = np.linalg.eigh(cov)
    evals = evals[::-1]
    lambd = evals[:rank]
    evecs = evecs[:, ::-1]
    q_mat = evecs[:, :rank]
    return {"mu": mu, "lambd": lambd, "q_mat": q_mat}
