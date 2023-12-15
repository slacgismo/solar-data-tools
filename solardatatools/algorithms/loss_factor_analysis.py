""" Combined Soiling and Degradation Estimation Module

This module is for estimation of degradation and soiling losses from unlabeled
daily energy production data. Model is of the form

y_t = x_t * d_t * s_t * w_t, for t = 1,...,T

where y_t [kWh] is the measured real daily energy on each day, x_t [kWh] is an ideal yearly baseline of performance,
and d_t, s_t, and w_t are the loss factors for degradation, soiling, and weather respectively.

"""

from gfosd import Problem
import gfosd.components as comp
from spcqe.functions import make_basis_matrix, make_regularization_matrix


class LossFactorEstimator:
    def __init__(self, energy_data, **kwargs):
        self.energy_data = energy_data
        log_energy = np.zeros_like(self.energy_data)
        is_zero = np.isclose(dh.daily_signals.energy, 0, atol=1e-1)
        log_energy[is_zero] = np.nan
        log_energy[~is_zero] = np.log(dh.daily_signals.energy[~is_zero])
        self.log_energy = log_energy
        self.use_ixs = ~is_zero
        self.problem = self.make_problem(**kwargs)
        self.degradation_rate = None
        self.energy_model = None
        self.log_energy_model = None
        self.total_energy_loss = None
        self.total_percent_loss = None
        self.degradation_energy_loss = None
        self.degradation_percent_loss = None
        self.soiling_energy_loss = None
        self.soiling_percent_loss = None
        self.weather_energy_loss = None
        self.weather_percent_loss = None

    def estimate_losses(self, solver="CLARABEL"):
        self.problem.decompose(solver=solver)
        # in the SD formulation, we put the residual term first, so it's the reverse order of how we specify this model (weather last)
        self.log_energy_model = self.problem.decomposition[::-1]
        self.energy_model = np.exp(self.log_energy_model)
        self.degradation_rate = 100 * np.median(
            (self.energy_model[1][365:] - self.energy_model[1][:-365])
            / self.energy_model[1][365:]
        )
        total_energy = np.sum(self.energy_data[self.use_ixs])
        self.total_energy_loss = total_energy - np.sum(
            self.energy_model[0][self.use_ixs]
        )
        self.degradation_energy_loss = total_energy - np.sum(
            np.product(self.energy_model[[True, False, True, True]], axis=0)[
                self.use_ixs
            ]
        )
        self.soiling_energy_loss = total_energy - np.sum(
            np.product(self.energy_model[[True, True, False, True]], axis=0)[
                self.use_ixs
            ]
        )
        self.weather_energy_loss = total_energy - np.sum(
            np.product(self.energy_model[[True, True, True, False]], axis=0)[
                self.use_ixs
            ]
        )
        items = [
            self.degradation_energy_loss,
            self.soiling_energy_loss,
            self.weather_energy_loss,
        ]
        avg_item = np.average(items)
        new_items = []
        for item in items:
            item -= avg_item
            item += self.total_energy_loss / len(items)
            new_items.append(item)
        (
            self.degradation_energy_loss,
            self.soiling_energy_loss,
            self.weather_energy_loss,
        ) = new_items
        self.total_percent_loss = (
            100 * self.total_energy_loss / np.sum(self.energy_data)
        )
        self.degradation_percent_loss = (
            100 * self.degradation_energy_loss / np.sum(self.energy_data)
        )
        self.soiling_percent_loss = (
            100 * self.soiling_energy_loss / np.sum(self.energy_data)
        )
        self.weather_percent_loss = (
            100 * self.weather_energy_loss / np.sum(self.energy_data)
        )

        return

    def report(self):
        if self.total_energy_loss is not None:
            out = {
                "degradation rate [%/yr]": self.degradation_rate,
                "total percent loss [%]": self.total_percent_loss,
                "degradation percent loss [%]": self.degradation_percent_loss,
                "soiling percent loss [%]": self.soiling_percent_loss,
                "weather percent loss [%]": self.weather_percent_loss,
                "total energy loss [kWh]": self.total_energy_loss,
                "degradation energy loss [kWh]": self.degradation_energy_loss,
                "soiling energy loss [kWh]": self.soiling_energy_loss,
                "weather energy loss [kWh]": self.weather_energy_loss,
            }
            return out

    def holdout_validate(self, seed=None, solver="CLARABEL"):
        residual, test_ix = self.problem.holdout_decompose(seed=seed, solver=solver)
        error_metric = np.sum(np.abs(residual))
        return error_metric

    def make_problem(
        self,
        tau=0.95,
        num_harmonics=4,
        deg_type="linear",
        include_soiling=True,
        weight_seasonal=10e-2,
        weight_soiling_stiffness=1e1,
        weight_soiling_sparsity=1e-2,
        weight_deg_nonlinear=10e4,
    ):
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
                    comp.SumAbs(weight=weight_soiling_sparsity),
                ]
            )
        else:
            c3 = comp.Aggregate([comp.NoSlope(), comp.FirstValEqual(value=0)])
        # Degradation term
        if deg_type == "linear":
            c4 = comp.Aggregate([comp.NoCurvature(), comp.FirstValEqual(value=0)])
        elif deg_type == "nonlinear":
            n_tot = length
            n_reduce = int(0.9 * n_tot)
            bottom_mat = sp.lil_matrix((n_tot - n_reduce, n_reduce))
            bottom_mat[:, -1] = 1
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
            c4 = comp.Aggregate([comp.NoSlope(), comp.FirstValEqual(value=0)])

        prob = Problem(self.log_energy, [c1, c3, c4, c2])
        return prob
