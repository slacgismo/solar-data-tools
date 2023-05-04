"""
This module defines "Statistical Clear Sky Fitting" algorithm.
"""

from time import time
import numpy as np
from numpy.linalg import norm
import cvxpy as cvx
from collections import defaultdict
from statistical_clear_sky.algorithm.initialization.singular_value_decomposition import (
    SingularValueDecomposition,
)
from statistical_clear_sky.algorithm.initialization.linearization_helper import (
    LinearizationHelper,
)
from statistical_clear_sky.algorithm.initialization.weight_setting import WeightSetting
from statistical_clear_sky.algorithm.exception import ProblemStatusError
from statistical_clear_sky.algorithm.minimization.left_matrix import (
    LeftMatrixMinimization,
)
from statistical_clear_sky.algorithm.minimization.right_matrix import (
    RightMatrixMinimization,
)
from statistical_clear_sky.algorithm.serialization.state_data import StateData
from statistical_clear_sky.algorithm.serialization.serialization_mixin import (
    SerializationMixin,
)
from statistical_clear_sky.algorithm.plot.plot_mixin import PlotMixin
from statistical_clear_sky.utilities.data_loading import resample_index
from statistical_clear_sky.utilities.progress import progress


class IterativeFitting(SerializationMixin, PlotMixin):
    """
    Implementation of "Statistical Clear Sky Fitting" algorithm.
    """

    def __init__(
        self,
        data_matrix=None,
        data_handler_obj=None,
        rank_k=6,
        solver_type="MOSEK",
        reserve_test_data=False,
    ):
        """

        :param data_matrix:
        :param data_handler_obj:
        :param rank_k:
        :param solver_type:
        :param reserve_test_data:
        """
        self._solver_type = solver_type
        self._rank_k = rank_k
        if data_handler_obj is None and data_matrix is None:
            print("Please initialize class with a data set")
        elif data_handler_obj is not None:
            data_matrix = data_handler_obj.filled_data_matrix
            self._power_signals_d = data_matrix
            self._capacity = data_handler_obj.capacity_estimate
            # Set the weighting now, to use the error flagging feature
            weights = self._get_weight_setting().obtain_weights(data_matrix)
            weights *= data_handler_obj.daily_flags.no_errors
        else:
            self._power_signals_d = data_matrix

        self._decomposition = SingularValueDecomposition()
        self._decomposition.decompose(data_matrix, rank_k=rank_k)

        self._matrix_l0 = self._decomposition.matrix_l0
        self._matrix_r0 = self._decomposition.matrix_r0
        self._bootstrap_samples = None

        self._set_testdays(data_matrix, reserve_test_data)
        # Handle both DataHandler objects and reserving test data
        if data_handler_obj is not None and self._test_days is not None:
            weights[self._test_days] = 0

        # Stores the current state of the object:
        self._state_data = StateData()
        self._store_initial_state_data()
        if data_handler_obj is not None:
            self._weights = weights
            self._state_data.weights = weights

        self._set_residuals()

    def execute(
        self,
        mu_l=None,
        mu_r=None,
        tau=None,
        exit_criterion_epsilon=1e-3,
        max_iteration=10,
        is_degradation_calculated=True,
        max_degradation=None,
        min_degradation=None,
        non_neg_constraints=False,
        verbose=True,
        bootstraps=None,
    ):

        mu_l, mu_r, tau = self._obtain_hyper_parameters(mu_l, mu_r, tau)
        l_cs_value, r_cs_value, beta_value = self._obtain_initial_values()
        weights = self._obtain_weights(verbose=verbose)
        component_r0 = self._obtain_initial_component_r0(verbose=verbose)
        self.__left_first = True

        self._minimize_objective(
            l_cs_value,
            r_cs_value,
            beta_value,
            component_r0,
            weights,
            mu_l=mu_l,
            mu_r=mu_r,
            tau=tau,
            exit_criterion_epsilon=exit_criterion_epsilon,
            max_iteration=max_iteration,
            is_degradation_calculated=is_degradation_calculated,
            max_degradation=max_degradation,
            min_degradation=min_degradation,
            non_neg_constraints=non_neg_constraints,
            verbose=verbose,
            bootstraps=bootstraps,
        )

        self._keep_supporting_parameters_as_properties(weights)
        self._store_final_state_data(weights)

    def calculate_objective_with_result(self, sum_components=True):
        return self._calculate_objective(
            self._state_data.mu_l,
            self._state_data.mu_r,
            self._state_data.tau,
            self._l_cs_value,
            self._r_cs_value,
            self._beta_value,
            self._weights,
            sum_components=sum_components,
        )

    @property
    def measured_power_matrix(self):
        return self._power_signals_d

    @property
    def estimated_power_matrix(self):
        left = self._l_cs_value
        right = self._r_cs_value
        mat = left @ right
        return mat

    @property
    def estimated_clear_sky(self):
        return self._l_cs_value @ self._r_cs_value

    @property
    def deg_rate(self):
        return self._beta_value.item()

    @property
    def left_matrix(self):
        return self._l_cs_value

    @property
    def right_matrix(self):
        return self._r_cs_value

    @property
    def left_problem(self):
        return self._l_problem

    @property
    def right_problem(self):
        return self._r_problem

    @property
    def l_cs_value(self):
        return self._l_cs_value

    @property
    def r_cs_value(self):
        return self._r_cs_value

    @property
    def beta_value(self):
        return self._beta_value

    @property
    def weights(self):
        return self._weights

    @property
    def residuals_median(self):
        return self._residuals_median

    @property
    def residuals_variance(self):
        return self._residuals_variance

    @property
    def residual_l0_norm(self):
        return self._residual_l0_norm

    @property
    def fixed_time_stamps(self):
        return self._fixed_time_stamps

    @property
    def test_days(self):
        return self._test_days

    @property
    def state_data(self):
        return self._state_data

    @property
    def bootstrap_samples(self):
        return self._bootstrap_samples

    # Alias method for l_cs_value accessor (with property decorator):
    def left_low_rank_matrix(self):
        return self.l_cs_value

    # Alias method for r_cs_value accessor (with property decorator):
    def right_low_rank_matrix(self):
        return self.r_cs_value

    # Alias method for beta_value accessor (with property decorator):
    def degradation_rate(self):
        return self.beta_value

    def clear_sky_signals(self):
        return self._l_cs_value @ self._r_cs_value

    def _minimize_objective(
        self,
        l_cs_value,
        r_cs_value,
        beta_value,
        component_r0,
        weights,
        mu_l=None,
        mu_r=None,
        tau=None,
        exit_criterion_epsilon=1e-3,
        max_iteration=100,
        is_degradation_calculated=True,
        max_degradation=None,
        min_degradation=None,
        non_neg_constraints=True,
        verbose=True,
        bootstraps=None,
    ):
        left_matrix_minimization = self._get_left_matrix_minimization(
            weights, tau, mu_l, non_neg_constraints=non_neg_constraints
        )
        right_matrix_minimization = self._get_right_matrix_minimization(
            weights,
            tau,
            mu_r,
            non_neg_constraints=non_neg_constraints,
            is_degradation_calculated=is_degradation_calculated,
            max_degradation=max_degradation,
            min_degradation=min_degradation,
        )

        self._l_problem = left_matrix_minimization
        self._r_problem = right_matrix_minimization
        ti = time()
        objective_values = self._calculate_objective(
            mu_l,
            mu_r,
            tau,
            l_cs_value,
            r_cs_value,
            beta_value,
            weights,
            sum_components=False,
        )
        if verbose:
            print("----------------------\nSCSF Problem Setup\n----------------------")
            msg1 = "Matrix Size: {} x {} = {} power measurements".format(
                self._power_signals_d.shape[0],
                self._power_signals_d.shape[1],
                self._power_signals_d.size,
            )
            print(msg1)
            reduced_mat = self._power_signals_d[:, self._weights > 0]
            try:
                real_meas = reduced_mat > 0.005 * self._capacity
            except:
                real_meas = reduced_mat > 0.005 * np.nanquantile(
                    self._power_signals_d, 0.95
                )
            msg = "Sparsity: {:.2f}%".format(
                100 * (1 - np.sum(real_meas) / self._power_signals_d.size)
            )
            print(msg)
            msg = "{} non-zero measurements under clear conditions".format(
                np.sum(real_meas)
            )
            print(msg)
            msg2 = "Model size: {} x {} + {} x {} = {} parameters".format(
                l_cs_value.shape[0],
                l_cs_value.shape[1],
                r_cs_value.shape[0],
                r_cs_value.shape[1],
                np.sum(
                    [
                        l_cs_value.shape[0] * l_cs_value.shape[1],
                        r_cs_value.shape[0] * r_cs_value.shape[1],
                    ]
                ),
            )
            print(msg2)
            print("\n")
            print(
                "----------------------\nAlgorithm Iterations\n----------------------"
            )
            ps = "Starting at Objective: {:.3e}, f1: {:.3e}, f2: {:.3e},"
            ps += " f3: {:.3e}, f4: {:.3e}"
            print(
                ps.format(
                    np.sum(objective_values),
                    objective_values[0],
                    objective_values[1],
                    objective_values[2],
                    objective_values[3],
                )
            )
        improvement = np.inf
        old_objective_value = np.sum(objective_values)
        iteration = 0
        f1_last = objective_values[0]

        tol_schedule = []  # np.logspace(-4, -8, 6)

        while improvement >= exit_criterion_epsilon:
            try:
                tol = tol_schedule[iteration]
            except IndexError:
                tol = 1e-8

            self._store_minimization_state_data(
                mu_l, mu_r, tau, l_cs_value, r_cs_value, beta_value, component_r0
            )

            try:
                if self.__left_first:
                    if verbose:
                        print("    Minimizing left matrix")
                    (
                        l_cs_value,
                        r_cs_value,
                        beta_value,
                    ) = left_matrix_minimization.minimize(
                        l_cs_value, r_cs_value, beta_value, component_r0, tol=tol
                    )
                    if verbose:
                        print("    Minimizing right matrix")
                    (
                        l_cs_value,
                        r_cs_value,
                        beta_value,
                    ) = right_matrix_minimization.minimize(
                        l_cs_value, r_cs_value, beta_value, component_r0, tol=tol
                    )
                else:
                    if verbose:
                        print("    Minimizing right matrix")
                    (
                        l_cs_value,
                        r_cs_value,
                        beta_value,
                    ) = right_matrix_minimization.minimize(
                        l_cs_value, r_cs_value, beta_value, component_r0, tol=tol
                    )
                    if verbose:
                        print("    Minimizing left matrix")
                    (
                        l_cs_value,
                        r_cs_value,
                        beta_value,
                    ) = left_matrix_minimization.minimize(
                        l_cs_value, r_cs_value, beta_value, component_r0, tol=tol
                    )
            except cvx.SolverError:
                if self.__left_first:
                    if verbose:
                        print(
                            "Solver failed! Starting over and reversing minimization order."
                        )
                    self.__left_first = False
                    iteration = 0
                    l_cs_value = self._decomposition.matrix_l0
                    r_cs_value = self._decomposition.matrix_r0
                    component_r0 = self._obtain_initial_component_r0(verbose=verbose)
                    continue
                else:
                    if verbose:
                        print("Solver failing again! Exiting...")
                    self._state_data.is_solver_error = True
                    break
            except ProblemStatusError as e:
                if verbose:
                    print(e)
                if self.__left_first:
                    if verbose:
                        print("Starting over and reversing minimization order.")
                    self.__left_first = False
                    iteration = 0
                    l_cs_value = self._decomposition.matrix_l0
                    r_cs_value = self._decomposition.matrix_r0
                    component_r0 = self._obtain_initial_component_r0(verbose=verbose)
                    continue
                else:
                    if verbose:
                        print("Exiting...")
                    self._state_data.is_problem_status_error = True
                    break

            component_r0 = r_cs_value[0, :]

            objective_values = self._calculate_objective(
                mu_l,
                mu_r,
                tau,
                l_cs_value,
                r_cs_value,
                beta_value,
                weights,
                sum_components=False,
            )
            new_objective_value = np.sum(objective_values)
            improvement = (
                (old_objective_value - new_objective_value) * 1.0 / old_objective_value
            )
            old_objective_value = new_objective_value
            iteration += 1
            if verbose:
                ps = "{} - Objective: {:.3e}, f1: {:.3e}, f2: {:.3e},"
                ps += " f3: {:.3e}, f4: {:.3e}"
                print(
                    ps.format(
                        iteration,
                        new_objective_value,
                        objective_values[0],
                        objective_values[1],
                        objective_values[2],
                        objective_values[3],
                    )
                )
            if objective_values[0] > f1_last:
                self._state_data.f1_increase = True
                if verbose:
                    print("Caution: residuals increased")
            if improvement < 0:
                if verbose:
                    print("Caution: objective increased.")
                self._state_data.obj_increase = True
                improvement *= -1
            if objective_values[3] > 1e2:
                if self.__left_first:
                    if verbose:
                        print(
                            "Bad trajectory detected. Starting over and reversing minimization order."
                        )
                    self.__left_first = False
                    iteration = 0
                    l_cs_value = self._decomposition.matrix_l0
                    r_cs_value = self._decomposition.matrix_r0
                    component_r0 = self._obtain_initial_component_r0(verbose=verbose)
                else:
                    if verbose:
                        print("Algorithm Failed!")
                    improvement = 0
            if iteration >= max_iteration:
                if verbose:
                    print(
                        "Reached iteration limit. Previous improvement: {:.2f}%".format(
                            improvement * 100
                        )
                    )
                improvement = 0.0

            self._store_minimization_state_data(
                mu_l, mu_r, tau, l_cs_value, r_cs_value, beta_value, component_r0
            )

        # except cvx.SolverError:
        #     if self.__left_first:
        #         if verbose:
        #             print('solver failed! Starting over and reversing minimization order.')
        #
        #     self._state_data.is_solver_error = True
        # except ProblemStatusError as e:
        #     if verbose:
        #         print(e)
        #     self._state_data.is_problem_status_error = True

        tf = time()
        if verbose:
            print("Minimization complete in {:.2f} minutes".format((tf - ti) / 60.0))
        self._analyze_residuals(l_cs_value, r_cs_value, weights)
        self._keep_result_variables_as_properties(l_cs_value, r_cs_value, beta_value)
        if bootstraps is not None:
            if verbose:
                print("Running bootstrap analysis...")
            ti = time()
            self._bootstrap_samples = defaultdict(dict)
            for ix in range(bootstraps):
                # resample the days with non-zero weights only
                bootstrap_weights = resample_index(length=np.sum(weights > 1e-1))
                new_weights = np.zeros_like(weights)
                new_weights[weights > 1e-1] = bootstrap_weights
                new_weights = np.multiply(weights, new_weights)
                left_matrix_minimization.update_weights(new_weights)
                right_matrix_minimization.update_weights(new_weights)
                l_cs_value = self._l_cs_value
                r_cs_value = self._r_cs_value
                beta_value = self._beta_value
                # ti = time()
                objective_values = self._calculate_objective(
                    mu_l,
                    mu_r,
                    tau,
                    l_cs_value,
                    r_cs_value,
                    beta_value,
                    new_weights,
                    sum_components=False,
                )
                if verbose:
                    progress(
                        ix,
                        bootstraps,
                        status=" {:.2f} minutes".format((time() - ti) / 60),
                    )
                    # ps = 'Bootstrap Sample {}\n'.format(ix)
                    # ps += 'Starting at Objective: {:.3e}, f1: {:.3e}, f2: {:.3e},'
                    # ps += ' f3: {:.3e}, f4: {:.3e}'
                    # print(ps.format(
                    #     np.sum(objective_values), objective_values[0],
                    #     objective_values[1], objective_values[2],
                    #     objective_values[3]
                    # ))
                improvement = np.inf
                old_objective_value = np.sum(objective_values)
                iteration = 0
                f1_last = objective_values[0]

                tol_schedule = []  # np.logspace(-4, -8, 6)

                while improvement >= exit_criterion_epsilon:
                    try:
                        tol = tol_schedule[iteration]
                    except IndexError:
                        tol = 1e-8

                    # self._store_minimization_state_data(mu_l, mu_r, tau,
                    #                                     l_cs_value, r_cs_value,
                    #                                     beta_value,
                    #                                     component_r0)

                    try:
                        if self.__left_first:
                            # if verbose:
                            # print('    Minimizing left matrix')
                            (
                                l_cs_value,
                                r_cs_value,
                                beta_value,
                            ) = left_matrix_minimization.minimize(
                                l_cs_value,
                                r_cs_value,
                                beta_value,
                                component_r0,
                                tol=tol,
                            )
                            # if verbose:
                            # print('    Minimizing right matrix')
                            (
                                l_cs_value,
                                r_cs_value,
                                beta_value,
                            ) = right_matrix_minimization.minimize(
                                l_cs_value,
                                r_cs_value,
                                beta_value,
                                component_r0,
                                tol=tol,
                            )
                        else:
                            # if verbose:
                            # print('    Minimizing right matrix')
                            (
                                l_cs_value,
                                r_cs_value,
                                beta_value,
                            ) = right_matrix_minimization.minimize(
                                l_cs_value,
                                r_cs_value,
                                beta_value,
                                component_r0,
                                tol=tol,
                            )
                            # if verbose:
                            # print('    Minimizing left matrix')
                            (
                                l_cs_value,
                                r_cs_value,
                                beta_value,
                            ) = left_matrix_minimization.minimize(
                                l_cs_value,
                                r_cs_value,
                                beta_value,
                                component_r0,
                                tol=tol,
                            )
                    except cvx.SolverError:
                        if self.__left_first:
                            if verbose:
                                print(
                                    "Solver failed! Starting over and reversing minimization order."
                                )
                            self.__left_first = False
                            iteration = 0
                            l_cs_value = self._decomposition.matrix_l0
                            r_cs_value = self._decomposition.matrix_r0
                            component_r0 = self._obtain_initial_component_r0(
                                verbose=verbose
                            )
                            continue
                        else:
                            if verbose:
                                print("Solver failing again! Exiting...")
                            self._state_data.is_solver_error = True
                            break
                    except ProblemStatusError as e:
                        if verbose:
                            print(e)
                        if self.__left_first:
                            if verbose:
                                print("Starting over and reversing minimization order.")
                            self.__left_first = False
                            iteration = 0
                            l_cs_value = self._decomposition.matrix_l0
                            r_cs_value = self._decomposition.matrix_r0
                            component_r0 = self._obtain_initial_component_r0(
                                verbose=verbose
                            )
                            continue
                        else:
                            if verbose:
                                print("Exiting...")
                            self._state_data.is_problem_status_error = True
                            break

                    component_r0 = r_cs_value[0, :]

                    objective_values = self._calculate_objective(
                        mu_l,
                        mu_r,
                        tau,
                        l_cs_value,
                        r_cs_value,
                        beta_value,
                        new_weights,
                        sum_components=False,
                    )
                    new_objective_value = np.sum(objective_values)
                    improvement = (
                        (old_objective_value - new_objective_value)
                        * 1.0
                        / old_objective_value
                    )
                    old_objective_value = new_objective_value
                    iteration += 1
                    # if verbose:
                    # ps = '{} - Objective: {:.3e}, f1: {:.3e}, f2: {:.3e},'
                    # ps += ' f3: {:.3e}, f4: {:.3e}'
                    # print(ps.format(
                    #     iteration, new_objective_value,
                    #     objective_values[0],
                    #     objective_values[1], objective_values[2],
                    #     objective_values[3]
                    # ))
                    if objective_values[0] > f1_last:
                        self._state_data.f1_increase = True
                        if verbose:
                            print("Caution: residuals increased")
                    if improvement < 0:
                        if verbose:
                            print("Caution: objective increased.")
                        self._state_data.obj_increase = True
                        improvement *= -1
                    if objective_values[3] > 1e2:
                        if self.__left_first:
                            if verbose:
                                print(
                                    "Bad trajectory detected. Starting over and reversing minimization order."
                                )
                            self.__left_first = False
                            iteration = 0
                            l_cs_value = self._decomposition.matrix_l0
                            r_cs_value = self._decomposition.matrix_r0
                            component_r0 = self._obtain_initial_component_r0(
                                verbose=verbose
                            )
                        else:
                            if verbose:
                                print("Algorithm Failed!")
                            improvement = 0
                    if iteration >= max_iteration:
                        if verbose:
                            print(
                                "Reached iteration limit. Previous improvement: {:.2f}%".format(
                                    improvement * 100
                                )
                            )
                        improvement = 0.0
                # tf = time()
                # if verbose:
                #     print('Bootstrap {} complete in {:.2f} minutes'.format(
                #           ix, (tf - ti) / 60.))
                self._bootstrap_samples[ix]["L"] = l_cs_value
                self._bootstrap_samples[ix]["R"] = r_cs_value
                self._bootstrap_samples[ix]["beta"] = beta_value
            if verbose:
                progress(
                    bootstraps,
                    bootstraps,
                    status=" {:.2f} minutes".format((time() - ti) / 60),
                )

    def _calculate_objective(
        self,
        mu_l,
        mu_r,
        tau,
        l_cs_value,
        r_cs_value,
        beta_value,
        weights,
        sum_components=True,
    ):
        weights_w1 = np.diag(weights)
        # Note: Not using cvx.sum and cvx.abs as in following caused
        # an error at * weights_w1:
        # ValueError: operands could not be broadcast together with shapes
        # (288,1300) (1300,1300)
        # term_f1 = sum((0.5 * abs(
        #     self._power_signals_d - l_cs_value.dot(r_cs_value))
        #     + (tau - 0.5)
        #     * (self._power_signals_d - l_cs_value.dot(r_cs_value)))
        #     * weights_w1)
        term_f1 = (
            cvx.sum(
                (
                    0.5 * cvx.abs(self._power_signals_d - l_cs_value @ r_cs_value)
                    + (tau - 0.5) * (self._power_signals_d - l_cs_value @ r_cs_value)
                )
                @ weights_w1
            )
        ).value
        weights_w2 = np.eye(self._rank_k)
        term_f2 = mu_l * norm(
            (l_cs_value[:-2, :] - 2 * l_cs_value[1:-1, :] + l_cs_value[2:, :])
            @ weights_w2,
            "fro",
        )
        term_f3 = mu_r * norm(
            r_cs_value[:, :-2] - 2 * r_cs_value[:, 1:-1] + r_cs_value[:, 2:], "fro"
        )
        if r_cs_value.shape[1] < 365 + 2:
            term_f4 = 0
        else:
            # Note: it was cvx.norm. Check if this modification makes a
            # difference:
            # term_f4 = (mu_r * norm(
            #             r_cs_value[1:, :-365] - r_cs_value[1:, 365:], 'fro'))
            term_f4 = (
                (mu_r * cvx.norm(r_cs_value[1:, :-365] - r_cs_value[1:, 365:], "fro"))
            ).value
        components = [term_f1, term_f2, term_f3, term_f4]
        objective = sum(components)
        if sum_components:
            return objective
        else:
            return components

    def _obtain_hyper_parameters(self, mu_l, mu_r, tau):
        if mu_l is None and self._state_data.mu_l is not None:
            mu_l = self._state_data.mu_l
        if mu_r is None and self._state_data.mu_r is not None:
            mu_r = self._state_data.mu_r
        if tau is None and self._state_data.tau is not None:
            tau = self._state_data.tau
        return mu_l, mu_r, tau

    def _obtain_initial_values(self):
        if self._state_data.l_value.size > 0:
            l_cs_value = self._state_data.l_value
        else:
            l_cs_value = self._decomposition.matrix_l0
        if self._state_data.r_value.size > 0:
            r_cs_value = self._state_data.r_value
        else:
            r_cs_value = self._decomposition.matrix_r0
        if self._state_data.beta_value != 0.0:
            beta_value = self._state_data.beta_value
        else:
            beta_value = 0.0
        self._keep_result_variables_as_properties(l_cs_value, r_cs_value, beta_value)
        return l_cs_value, r_cs_value, beta_value

    def _obtain_initial_component_r0(self, verbose=True):
        if verbose:
            # print('obtaining initial value of component r0')
            pass
        if self._state_data.component_r0.size > 0:
            # component_r0 = self._state_data.component_r0
            component_r0 = np.ones(self._decomposition.matrix_r0.shape[1])
        else:
            # component_r0 = self._get_linearization_helper().obtain_component_r0(
            #     self._decomposition.matrix_r0, index_set=self.weights > 1e-3)
            component_r0 = np.ones(self._decomposition.matrix_r0.shape[1])
        return component_r0

    def _obtain_weights(self, verbose=True):
        if verbose:
            # print('obtaining weights')
            pass
        if self._state_data.weights.size > 0:
            weights = self._state_data.weights
        else:
            weights = self._get_weight_setting().obtain_weights(self._power_signals_d)
            if self._test_days is not None:
                weights[self._test_days] = 0
        self._weights = weights
        return weights

    def _set_testdays(self, power_signals_d, reserve_test_data):
        if reserve_test_data:
            m, n = power_signals_d.shape
            day_indices = np.arange(n)
            num = int(n * reserve_test_data)
            self._test_days = np.sort(np.random.choice(day_indices, num, replace=False))
        else:
            self._test_days = None

    def _set_residuals(self):
        if self._state_data.residuals_median is not None:
            self._residuals_median = self._state_data.residuals_median
        else:
            self._residuals_median = None
        if self._state_data.residuals_variance is not None:
            self._residuals_variance = self._state_data.residuals_variance
        else:
            self._residuals_variance = None
        if self._state_data.residual_l0_norm is not None:
            self._residual_l0_norm = self._state_data.residual_l0_norm
        else:
            self._residual_l0_norm = None

    def _analyze_residuals(self, l_cs_value, r_cs_value, weights):
        # Residual analysis
        weights_w1 = np.diag(weights)
        wres = (l_cs_value @ r_cs_value - self._power_signals_d) @ weights_w1
        use_days = np.logical_not(np.isclose(np.sum(wres, axis=0), 0))
        scaled_wres = wres[:, use_days] / np.average(self._power_signals_d[:, use_days])
        final_metric = scaled_wres[self._power_signals_d[:, use_days] > 1e-3]
        self._residuals_median = np.median(final_metric)
        self._residuals_variance = np.power(np.std(final_metric), 2)
        self._residual_l0_norm = np.linalg.norm(
            self._matrix_l0[:, 0] - l_cs_value[:, 0]
        )

    def _get_linearization_helper(self):
        """
        For dependency injection for testing, i.e. for injecting mock.
        """
        if (not hasattr(self, "_linearization_helper")) or (
            self._linearization_helper is None
        ):
            return LinearizationHelper(solver_type=self._solver_type)
        else:  # This must be mock object inject from test
            return self._linearization_helper

    def set_linearization_helper(self, value):
        """
        For dependency injection for testing, i.e. for injecting mock.
        """
        self._linearization_helper = value

    def _get_weight_setting(self):
        """
        For dependency injection for testing, i.e. for injecting mock.
        """
        if (not hasattr(self, "_weight_setting")) or (self._weight_setting is None):
            return WeightSetting(solver_type=self._solver_type)
        else:  # This must be mock object inject from test
            return self._weight_setting

    def set_weight_setting(self, value):
        """
        For dependency injection for testing, i.e. for injecting mock.
        """
        self._weight_setting = value

    def _get_left_matrix_minimization(
        self, weights, tau, mu_l, non_neg_constraints=True
    ):
        """
        For dependency injection for testing, i.e. for injecting mock.
        """
        if (not hasattr(self, "_left_matrix_minimization")) or (
            self._left_matrix_minimization is None
        ):
            return LeftMatrixMinimization(
                self._power_signals_d,
                self._rank_k,
                weights,
                tau,
                mu_l,
                non_neg_constraints=non_neg_constraints,
                solver_type=self._solver_type,
            )
        else:  # This must be mock object inject from test
            return self._left_matrix_minimization

    def set_left_matrix_minimization(self, value):
        """
        For dependency injection for testing, i.e. for injecting mock.
        """
        self._left_matrix_minimization = value

    def _get_right_matrix_minimization(
        self,
        weights,
        tau,
        mu_r,
        non_neg_constraints=True,
        is_degradation_calculated=True,
        max_degradation=None,
        min_degradation=None,
    ):
        """
        For dependency injection for testing, i.e. for injecting mock.
        """
        if (not hasattr(self, "_right_matrix_minimization")) or (
            self._right_matrix_minimization is None
        ):
            return RightMatrixMinimization(
                self._power_signals_d,
                self._rank_k,
                weights,
                tau,
                mu_r,
                non_neg_constraints=non_neg_constraints,
                is_degradation_calculated=is_degradation_calculated,
                max_degradation=max_degradation,
                min_degradation=min_degradation,
                solver_type=self._solver_type,
            )
        else:  # This must be mock object inject from test
            return self._right_matrix_minimization

    def set_right_matrix_minimization(self, value):
        """
        For dependency injection for testing, i.e. for injecting mock.
        """
        self._right_matrix_minimization = value

    def _keep_result_variables_as_properties(self, l_cs_value, r_cs_value, beta_value):
        self._l_cs_value = l_cs_value
        self._r_cs_value = r_cs_value
        self._beta_value = beta_value

    def _keep_supporting_parameters_as_properties(self, weights):
        self._weights = weights

    def _store_initial_state_data(self):
        self._state_data.power_signals_d = self._power_signals_d
        self._state_data.rank_k = self._rank_k
        self._state_data.matrix_l0 = self._matrix_l0
        self._state_data.matrix_r0 = self._matrix_r0
        self._state_data.mu_l = 5e2
        self._state_data.mu_r = 1e3
        self._state_data.tau = 0.85

    def _store_minimization_state_data(
        self, mu_l, mu_r, tau, l_cs_value, r_cs_value, beta_value, component_r0
    ):
        self._state_data.mu_l = mu_l
        self._state_data.mu_r = mu_r
        self._state_data.tau = tau
        self._state_data.l_value = l_cs_value
        self._state_data.r_value = r_cs_value
        self._state_data.beta_value = beta_value
        self._state_data.component_r0 = component_r0

    def _store_final_state_data(self, weights):
        self._state_data.residuals_median = self._residuals_median
        self._state_data.residuals_variance = self._residuals_variance
        self._state_data.residual_l0_norm = self._residual_l0_norm
        self._state_data.weights = weights
