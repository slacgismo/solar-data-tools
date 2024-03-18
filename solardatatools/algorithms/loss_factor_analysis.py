""" Combined Soiling and Degradation Estimation Module

This module is for estimation of degradation and soiling losses from unlabeled
daily energy production data. Model is of the form

y_t = x_t * d_t * s_t * c_t * w_t, for t \in K

where y_t [kWh] is the measured real daily energy on each day, x_t [kWh] is an ideal yearly baseline of performance,
and d_t, s_t, and w_t are the loss factors for degradation, soiling, capacity changes, and weather respectively. K is
the set of "known" index values, e.g. the days that we have good energy production values for (not missing or
corrupted).

Author: Bennet Meyers
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gfosd import Problem
import gfosd.components as comp
from gfosd.components.base_graph_class import GraphComponent
from spcqe.functions import make_basis_matrix, make_regularization_matrix
from tqdm import tqdm


class LossFactorAnalysis:
    def __init__(
        self, energy_data, capacity_change_labels=None, outage_flags=None, **kwargs
    ):
        """
        A class for analyzing loss factors, including the degradation rate, of a PV system

        :param energy_data: a discrete time series of daily integrated energy
        :type energy_data: numpy 1D array
        :param capacity_change_labels: labeling of known capacity changes, e.g. from the CapacityChange tool
        :type  capacity_change_labels: 1D array of integer labels, with same length as energy_data
        :param outage_flags: labeling of days with known operational issues, e.g. from DataHandle pipeline
        :type outage_flags: 1D array of integer labels, with same length as energy_data
        :param kwargs: arguments to be passed to to the make_problem method
        """
        self.energy_data = energy_data
        if len(self.energy_data) <= 365:
            print(
                "WARNING: This technique is not designed for use on less than one year of data!"
            )
        log_energy = np.zeros_like(self.energy_data)
        is_zero = np.isclose(energy_data, 0, atol=1e-1)
        log_energy[is_zero] = np.nan
        log_energy[~is_zero] = np.log(energy_data[~is_zero])
        self.log_energy = log_energy
        self.use_ixs = ~is_zero
        if outage_flags is not None:
            self.use_ixs = np.logical_and(self.use_ixs, ~outage_flags)
        self.capacity_change_labels = capacity_change_labels
        self.total_measured_energy = np.sum(self.energy_data[self.use_ixs])
        self.MC_results = None
        self.problem = self.make_problem(**kwargs)
        self.user_settings = kwargs
        self.degradation_rate = None
        self.degradation_rate_lb = None
        self.degradation_rate_ub = None
        self.energy_model = None
        self.log_energy_model = None
        self.total_energy_loss = None
        self.total_percent_loss = None
        self.degradation_energy_loss = None
        self.soiling_energy_loss = None
        self.capacity_change_loss = None
        self.weather_energy_loss = None
        self.outage_energy_loss = None
        self.degradation_percent = None
        self.soiling_percent = None
        self.capacity_change_percent = None
        self.weather_percent = None
        self.outage_percent = None

    def estimate_degradation_rate(
        self,
        max_samples=500,
        median_tol=5e-3,
        confidence_tol=1e-2,
        fraction_hold=0.2,
        method="median_unbiased",
        verbose=False,
    ):
        """
        This function runs a Monte Carlo simulation to estimate the uncertainty in the estimation of the degrdation rate
        based on the loss model. This will randomly sample problem parameters (quantile level and soiling stiffness
        weight), while randomly holding out 20% of the days each time. The algorithm exits when the estimates of the
        median, 2.5 percentile, and 97.5 percentile have stabilized. Results are stored in the following class
        attributes:
            self.degradation_rate
            self.degradation_rate_lb
            self.degradation_rate_ub
            self.MC_results

        :param max_samples: maximimum number of MC samples to generate (typically exits before this)
        :param median_tol: tolerance for median estimate stability
        :param confidence_tol: tolerance for outer percentile estimate stability
        :param fraction_hold: fraction of values to holdout in each sample
        :param method: quantile estimation method (see: https://numpy.org/doc/stable/reference/generated/numpy.quantile.html)
        :param verbose: control print statements
        :return: None
        """
        old_ixs = np.copy(self.use_ixs)
        change_is_small_now = False
        change_is_small_window = False
        output = pd.DataFrame(columns=["tau", "weight", "deg"])
        running_stats = pd.DataFrame(columns=["p50", "p025", "p975"])
        counter = 0
        if verbose:
            print(
                """
            ************************************************
            * Solar Data Tools Degradation Estimation Tool *
            ************************************************

            Monte Carlo sampling to generate a distributional estimate
            of the degradation rate [%/yr]

            The distribution typically stabilizes in 50-100 samples.

            Author: Bennet Meyers, SLAC

            This material is based upon work supported by the U.S. Department
            of Energy's Office of Energy Efficiency and Renewable Energy (EERE)
            under the Solar Energy Technologies Office Award Number 38529.\n
            """
            )
            progress = tqdm()
        while not (change_is_small_now and change_is_small_window):
            if verbose:
                progress.update()
            tau = np.random.uniform(0.85, 0.95)
            weight = np.random.uniform(0.5, 5)
            good_ixs = np.arange(len(self.use_ixs))[self.use_ixs]
            num_values = int(len(good_ixs) * fraction_hold)
            # select random index locations for removal
            selected_ixs = np.random.choice(good_ixs, size=num_values, replace=False)
            msk = np.zeros(len(self.use_ixs), dtype=bool)
            msk[selected_ixs] = True
            new_ixs = ~np.logical_or(~self.use_ixs, msk)
            self.use_ixs = new_ixs
            # remake modified problem
            self.problem = self.make_problem(tau=tau, weight_soiling_stiffness=weight)
            # run signal decomposition and shapley attribution
            self.estimate_losses()
            # record running results
            output.loc[counter] = [tau, weight, self.degradation_rate]
            running_stats.loc[counter] = [
                np.median(output["deg"]),
                np.quantile(output["deg"], 0.025, method=method),
                np.quantile(output["deg"], 0.975, method=method),
            ]
            self.use_ixs = old_ixs
            counter += 1
            diffs = np.diff(running_stats, axis=0)
            if verbose and (counter + 1) % 10 == 0:
                vn = running_stats.values[-1]
                progress.write(
                    f"P50, P02.5, P97.5: {vn[0]:.3f}, {vn[1]:.3f}, {vn[2]:.3f}"
                )
                progress.write(
                    f"changes: {diffs[-1][0]:.3e}, {diffs[-1][1]:.3e}, {diffs[-1][2]:.3e}"
                )
            # check exit conditions
            if counter < 20:
                # get at least 20 samples
                continue
            elif counter > max_samples:
                # don't go over max_samples
                break

            change_is_small_now = np.all(
                np.abs(diffs[-1]) <= [median_tol, confidence_tol, confidence_tol]
            )
            change_is_small_window = np.all(
                np.average(np.abs(diffs[-10:]), axis=0)
                <= [median_tol, confidence_tol, confidence_tol]
            )
        self.degradation_rate = np.median(output["deg"])
        self.degradation_rate_lb = np.quantile(output["deg"], 0.025, method=method)
        self.degradation_rate_ub = np.quantile(output["deg"], 0.975, method=method)
        self.MC_results = {"samples": output, "running stats": running_stats}
        self.problem = self.make_problem(**self.user_settings)

    def estimate_losses(self, solver="CLARABEL", verbose=False):
        self.problem.decompose(solver=solver, verbose=verbose)
        # in the SD formulation, we put the residual term first, so it's the reverse order of how we specify this model (weather last)
        self.log_energy_model = self.problem.decomposition[::-1]
        self.energy_model = np.exp(self.log_energy_model)
        self.degradation_rate = 100 * np.median(
            (self.energy_model[1][365:] - self.energy_model[1][:-365])
            / self.energy_model[1][365:]
        )
        # self.energy_lost_outages = np.sum(self.energy_model[:, self.use_ixs]) - np.sum(self.energy_model)
        total_energy = np.sum(self.energy_data[self.use_ixs])
        baseline_energy = np.sum(self.energy_model[0])
        self.total_energy_loss = total_energy - baseline_energy
        self.total_percent_loss = 100 * self.total_energy_loss / baseline_energy

        out = attribute_losses(self.energy_model, self.use_ixs)
        self.degradation_energy_loss = out[0]
        self.soiling_energy_loss = out[1]
        self.capacity_change_loss = out[2]
        self.weather_energy_loss = out[3]
        self.outage_energy_loss = out[4]

        self.degradation_percent = out[0] / self.total_energy_loss
        self.soiling_percent = out[1] / self.total_energy_loss
        self.capacity_change_percent = out[2] / self.total_energy_loss
        self.weather_percent = out[3] / self.total_energy_loss
        self.outage_percent = out[4] / self.total_energy_loss

        assert np.isclose(
            self.total_energy_loss,
            self.degradation_energy_loss
            + self.soiling_energy_loss
            + self.capacity_change_loss
            + self.weather_energy_loss
            + self.outage_energy_loss,
        )
        return

    def report(self):
        """
        Creates a machine-readible dictionary of result from the loss factor analysis
        :return: dictionary
        """
        if self.total_energy_loss is not None:
            out = {
                "degradation rate [%/yr]": self.degradation_rate,
                "deg rate lower bound [%/yr]": self.degradation_rate_lb,
                "deg rate upper bound [%/yr]": self.degradation_rate_ub,
                "total energy loss [kWh]": self.total_energy_loss,
                "degradation energy loss [kWh]": self.degradation_energy_loss,
                "soiling energy loss [kWh]": self.soiling_energy_loss,
                "capacity change energy loss [kWh]": self.capacity_change_loss,
                "weather energy loss [kWh]": self.weather_energy_loss,
                "system outage loss [kWh]": self.outage_energy_loss,
            }
            return out

    def plot_pie(self, figsize=None):
        """
        Create a pie plot of losses

        :return: matplotlib figure
        """
        if figsize is not None:
            plt.figure(figsize=figsize)
        plt.pie(
            [
                np.clip(-self.degradation_energy_loss, 0, np.inf),
                np.clip(-self.soiling_energy_loss, 0, np.inf),
                np.clip(-self.capacity_change_loss, 0, np.inf),
                np.clip(-self.weather_energy_loss, 0, np.inf),
                np.clip(-self.outage_energy_loss, 0, np.inf),
            ],
            labels=[
                "degradation",
                "soiling",
                "capacity change",
                "weather",
                "outages",
            ],
            autopct="%1.1f%%",
        )
        plt.title("System loss breakdown")
        return plt.gcf()

    def plot_waterfall(self):
        """
        Create a waterfall plot of losses

        :return: matplotlib figure
        """
        index = [
            "baseline",
            "weather",
            "outages",
            "capacity changes",
            "soiling",
            "degradation",
        ]
        bl = np.sum(self.energy_model[0])
        data = {
            "amount": [
                bl,
                self.weather_energy_loss,
                self.outage_energy_loss,
                self.capacity_change_loss,
                self.soiling_energy_loss,
                self.degradation_energy_loss,
            ]
        }
        fig = waterfall_plot(data, index)
        return fig

    def plot_decomposition(self, figsize=(16, 8.5)):
        """
        Creates a figure with subplots illustrating the estimated signal components found through decomposition

        :param figsize: size of figure (tuple)
        :return: matplotlib figure
        """
        _fig_decomp = self.problem.plot_decomposition(
            exponentiate=True, figsize=figsize
        )
        _ax = _fig_decomp.axes
        _ax[0].plot(
            np.arange(len(self.energy_data))[~self.use_ixs],
            self.energy_model[-1, ~self.use_ixs],
            color="red",
            marker=".",
            ls="none",
        )
        _ax[0].set_title("weather and system outages")
        _ax[1].set_title("capacity changes")
        _ax[2].set_title("soiling")
        _ax[3].set_title("degradation")
        _ax[4].set_title("baseline")
        _ax[5].set_title("measured energy (green) and model minus weather")
        plt.tight_layout()
        return _fig_decomp

    def plot_mc_histogram(self, figsize=None, title=None):
        """
        Creates a historgram of the Monte Carlo samples and annotates the chart with mean, median, mode, and confidence
        intervals.

        :param figsize: size of figure (tuple)
        :param title: title for figure (string)
        :return: matplotlib figure
        """
        if self.MC_results is not None:
            if figsize is not None:
                fig = plt.figure(figsize=figsize)
            degs = self.MC_results["samples"]["deg"]
            n, bins, patches = plt.hist(degs)
            plt.axvline(np.average(degs), color="yellow", label="mean")
            plt.axvline(np.median(degs), color="orange", label="median")
            mode_index = n.argmax()
            plt.axvline(
                bins[mode_index] + np.diff(bins)[0] / 2, color="red", label="mode"
            )
            plt.axvline(np.quantile(degs, 0.025), color="gray", ls=":")
            plt.axvline(
                np.quantile(degs, 0.975), color="gray", ls=":", label="95% confidence"
            )
            plt.legend()
            if title is not None:
                plt.title(title)
            return plt.gcf()

    def plot_mc_by_tau(self, figsize=None, title=None):
        """
        Creates a scatterplot of the Monte Carlo samples versus tau (quantile level) and colors the points by the
        weight of the soiling stiffness term


        :param figsize: size of figure (tuple)
        :param title: title for figure (string)
        :return: matplotlib figure
        """
        if self.MC_results is not None:
            if figsize is not None:
                fig = plt.figure(figsize=figsize)
            degs = self.MC_results["samples"]["deg"]
            taus = self.MC_results["samples"]["tau"]
            weights = self.MC_results["samples"]["weight"]
            plt.scatter(taus, degs, c=weights, cmap="plasma")
            plt.colorbar(label="soiling stiffness [1]")
            plt.xlabel("quantile level, tau [1]")
            plt.ylabel("degradation rate estimate [%/yr]")
            if title is not None:
                plt.title(title)
            return plt.gcf()

    def plot_mc_by_weight(self, figsize=None, title=None):
        """
        Creates a scatterplot of the Monte Carlo samples versus weight (soiling stiffness) and colors the points by the
        tau (quantile level)


        :param figsize: size of figure (tuple)
        :param title: title for figure (string)
        :return: matplotlib figure
        """
        if self.MC_results is not None:
            if figsize is not None:
                fig = plt.figure(figsize=figsize)
            degs = self.MC_results["samples"]["deg"]
            taus = self.MC_results["samples"]["tau"]
            weights = self.MC_results["samples"]["weight"]
            plt.scatter(weights, degs, c=taus, cmap="plasma")
            plt.colorbar(label="quantile level, tau [1]")
            plt.xlabel("soiling stiffness [1]")
            plt.ylabel("degradation rate estimate [%/yr]")
            if title is not None:
                plt.title(title)
            return plt.gcf()

    def make_problem(
        self,
        tau=0.9,
        num_harmonics=4,
        deg_type="linear",
        include_soiling=True,
        weight_seasonal=10e-2,
        weight_soiling_stiffness=1e0,
        weight_soiling_sparsity=1e-2,
        weight_deg_nonlinear=10e4,
        deg_rate=None,
    ):
        """
        Constuct the signal decomposition problem for estimation of loss factors in PV energy data.

        :param tau: the quantile level to fit
        :type tau: float
        :param num_harmonics: the number of harmonics to include in model for yearly periodicity
        :type num_harmonics: int
        :param deg_type: the type of degradation to model ("linear", "nonlinear", or "none")
        :type deg_type: str
        :param include_soiling: whether to include a soiling term
        :type include_soiling: bool
        :param weight_seasonal: the weight on the seasonal penalty term (higher is stiffer)
        :type weight_seasonal: float
        :param weight_soiling_stiffness: the weight on the soiling stiffness (higher is stiffer)
        :type weight_soiling_stiffness: float
        :param weight_soiling_sparsity: the weight on the soiling stiffness (higher is sparser)
        :type weight_soiling_sparsity: float
        :param weight_deg_nonlinear: only used if 'nonlinear' degradation model is selected
        :type weight_deg_nonlinear: float
        :param deg_rate: pass to set a known degradation rate rather than have the SD problem estimate it
        :type deg_rate: None or float [%/yr]
        :return: a gfosd.Problem instance
        """
        # Inherit degradation rate if Monte Carlo sampling has been conducted, but only if the user doesn't pass their
        # own rate
        if deg_rate is None and self.MC_results is not None:
            deg_rate = self.degradation_rate
        ### Construct Losses ###
        # NB: the order in which the lossses are defined here are *not* the order they are in the problem (see below)
        # Pinball loss noise
        c1 = comp.SumQuantile(tau=tau)
        # Smooth periodic term
        length = len(self.log_energy)
        periods = [365.2425]  # average length of a year in days
        _B = make_basis_matrix(num_harmonics, length, periods)
        _D = make_regularization_matrix(num_harmonics, weight_seasonal, periods)
        c2 = comp.Basis(basis=_B, penalty=_D)
        # Soiling term
        if include_soiling:
            c3 = comp.Aggregate(
                [
                    comp.Inequality(vmax=0),
                    comp.SumAbs(weight=weight_soiling_stiffness, diff=2),
                    comp.SumQuantile(
                        tau=0.98, weight=10 * weight_soiling_sparsity, diff=1
                    ),
                    comp.SumAbs(weight=weight_soiling_sparsity),
                ]
            )
        else:
            c3 = comp.Aggregate([comp.NoSlope(), comp.FirstValEqual(value=0)])
        # Degradation term
        if deg_type == "linear":
            if deg_rate is None:
                c4 = comp.Aggregate([comp.NoCurvature(), comp.FirstValEqual(value=0)])
            else:
                val = np.cumsum(np.r_[[0], np.ones(length - 1) * deg_rate / 100 / 365])
                c4 = SetEqual(val=val)
        elif deg_type == "nonlinear":
            n_tot = length
            n_reduce = int(0.9 * n_tot)
            bottom_mat = sp.lil_matrix((n_tot - n_reduce, n_reduce))
            bottom_mat[:, -1] = 1
            # don't allow any downward shifts in the last 10% of the data (constrain values to be equal)
            custom_basis = sp.bmat([[sp.eye(n_reduce)], [bottom_mat]])
            c4 = comp.Aggregate(
                [
                    comp.Inequality(vmax=0, diff=1),
                    comp.SumSquare(diff=2, weight=weight_deg_nonlinear),
                    comp.FirstValEqual(value=0),
                    comp.Basis(custom_basis),
                ]
            )
        elif deg_select.value == "none":
            c4 = SetEqual(val=np.zeros(length))
        # capacity change term — leverage previous analysis from SDT pipeline
        if self.capacity_change_labels is not None:
            basis_M = np.zeros((length, len(set(self.capacity_change_labels))))
            for lb in set(self.capacity_change_labels):
                slct = np.array(self.capacity_change_labels) == lb
                basis_M[slct, lb] = 1
            c5 = comp.Aggregate(
                [
                    comp.Inequality(vmax=0),
                    comp.Basis(basis=basis_M),
                    comp.SumAbs(weight=1e-6),
                ]
            )
        else:
            c5 = SetEqual(val=np.zeros(length))
        # Component order: weather, capacity change, soiling, degradation, baseline
        prob = Problem(self.log_energy, [c1, c5, c3, c4, c2], use_set=self.use_ixs)
        return prob

    def holdout_validate(self, seed=None, solver="CLARABEL"):
        residual, test_ix = self.problem.holdout_decompose(seed=seed, solver=solver)
        error_metric = np.sum(np.abs(residual))
        return error_metric


def model_wrapper(energy_model, use_ixs):
    n = energy_model.shape[0]

    def model_f(**kwargs):
        defaults = {f"arg{i + 1}": False for i in range(n)}
        defaults.update(kwargs)
        slct = [True] + [item for _, item in defaults.items()]
        apply_outages = slct[-1]
        slct = slct[:-1]
        model_select = energy_model[slct]
        daily_energy = np.product(model_select, axis=0)
        if apply_outages:
            daily_energy = daily_energy[use_ixs]
        return np.sum(daily_energy)

    return model_f


def enumerate_paths_full(origin, destination, path=None):
    """
    recursive algorithm for generating all possible monotonically increasing paths between
    two points on a n-dimensional hypercube
    """
    origin = list(origin)
    destination = list(destination)
    correct_ordering = np.all(
        np.asarray(destination, dtype=int) - np.asarray(origin, dtype=int) >= 0
    )
    if not correct_ordering:
        raise Exception("destination must be larger than origin in all dimensions")
    if path is None:
        path = []
    paths = []
    if origin == destination:
        # a path has been completed
        paths.append(path + [origin])
    else:
        # find the next index that can be incremented
        for i in range(len(origin)):
            if origin[i] != destination[i]:
                # create the next point in this path
                next_position = list(origin)
                next_position[i] = destination[0]
                # recurse to finish all paths that begin on this path
                paths.extend(
                    enumerate_paths_full(next_position, destination, path + [origin])
                )

    return paths


def enumerate_paths(n, dtype=int):
    """
    enumerates all possible paths from the origin to the ones vector in R^n
    """
    origin = np.zeros(n, dtype=dtype)
    destination = np.ones(n, dtype=dtype)
    return np.asarray(enumerate_paths_full(origin, destination))


def attribute_losses(energy_model, use_ixs):
    """This function assigns a total attribution to each loss factor, given a
    multiplicative loss factor model relative to a baseline, using Shapley
    attribution.

    :param energy_model: a multiplicative decomposition of PV daily energy, with the
        baseline first -- ie: baseline, degradation, soiling, capacity changes, and
        weather (residual)
    :type energy_model: 2d numpy array of shape n x T, where T is the number of days
        and n is the number of model factors
    :param use_ixs: a numpy boolean index where False records a system outage
    :type use_ixs: 1d numpy boolean array

    :return: a list of energy loss attributions, in the input order
    :rtype: 1d numpy float array
    """
    model_f = model_wrapper(energy_model, use_ixs)
    paths = enumerate_paths(energy_model.shape[0], dtype=bool)
    energy_estimates = np.zeros((paths.shape[0], paths.shape[1]))
    for ix, path in enumerate(paths):
        for jx, point in enumerate(path):
            kwargs = {f"arg{i + 1}": v for i, v in enumerate(point)}
            energy = model_f(**kwargs)
            energy_estimates[ix, jx] = energy
    lifts = np.diff(energy_estimates, axis=1)
    path_diffs = np.diff(paths, axis=1)
    ordering = np.argmax(path_diffs, axis=-1)
    ordered_lifts = np.take_along_axis(lifts, np.argsort(ordering, axis=1), axis=1)
    # print(energy_estimates)
    # print(lifts)
    # print(ordered_lifts)
    attributions = np.average(ordered_lifts, axis=0)
    total_energy = energy_estimates[0, -1]
    baseline_energy = energy_estimates[0, 0]
    # check that we've attributed all losses
    assert np.isclose(np.sum(attributions), total_energy - baseline_energy)
    return attributions


def waterfall_plot(data, index, figsize=(10, 4)):
    # Store data and create a blank series to use for the waterfall
    trans = pd.DataFrame(data=data, index=index)
    blank = trans.amount.cumsum().shift(1).fillna(0)

    # Get the net total number for the final element in the waterfall
    total = trans.sum().amount
    trans.loc["measured energy"] = total
    blank.loc["measured energy"] = total

    # The steps graphically show the levels as well as used for label placement
    step = blank.reset_index(drop=True).repeat(3).shift(-1)
    step[1::3] = np.nan

    # When plotting the last element, we want to show the full bar,
    # Set the blank to 0
    blank.loc["measured energy"] = 0

    # Plot and label
    my_plot = trans.plot(
        kind="bar",
        stacked=True,
        bottom=blank,
        legend=None,
        figsize=figsize,
        title="System Loss Factor Waterfall",
    )
    my_plot.plot(step.index, step.values, "k")
    my_plot.set_xlabel("Loss Factors")
    my_plot.set_ylabel("Energy (Wh)")

    # Get the y-axis position for the labels
    y_height = trans.amount.cumsum().shift(1).fillna(0)

    # Get an offset so labels don't sit right on top of the bar
    max = trans.max()
    max = max.iloc[0]
    neg_offset = max / 25
    pos_offset = max / 50
    plot_offset = int(max / 15)

    # Start label loop
    loop = 0
    for index, row in trans.iterrows():
        # For the last item in the list, we don't want to double count
        if row["amount"] == total:
            y = y_height.iloc[loop]
        else:
            y = y_height.iloc[loop] + row["amount"]
        # Determine if we want a neg or pos offset
        if row["amount"] > 0:
            y += pos_offset
        else:
            y -= neg_offset
        my_plot.annotate("{:,.0f}".format(row["amount"]), (loop, y), ha="center")
        loop += 1

    # Scale up the y axis so there is room for the labels
    my_plot.set_ylim(0, blank.max() + int(plot_offset))
    # Rotate the labels
    my_plot.set_xticklabels(trans.index, rotation=0)
    fig = my_plot.get_figure()
    fig.set_layout_engine(layout="tight")
    return fig


class SetEqual(GraphComponent):
    def __init__(self, val, *args, **kwargs):
        super().__init__(diff=0, *args, **kwargs)
        self._has_helpers = True
        self._val = val

    def _set_z_size(self):
        self._z_size = 0

    def _make_A(self):
        super()._make_A()

    def _make_c(self):
        self._c = self._val
