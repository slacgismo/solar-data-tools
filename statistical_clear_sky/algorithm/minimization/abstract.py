"""
This module defines common functionality of minimization problem solution
 process.
Since there is common code for minimization of both L matrix and R matrix,
the common code is placed in the abstract base class.
"""
from abc import abstractmethod
import cvxpy as cvx
import numpy as np

class AbstractMinimization():
    """
    Abstract class for minimization that uses the same equation but
    the subclasses fix either L (left) matrix value or R (right) matrix
    value.
    """

    def __init__(self, power_signals_d, rank_k, weights, tau,
                 non_neg_constraints=True, solver_type='ECOS'):
        self._power_signals_d = power_signals_d
        self._rank_k = rank_k
        self._weights = cvx.Parameter(shape=len(weights), value=weights,
                                      nonneg=True)
        self._tau = tau
        self._non_neg_constraints = non_neg_constraints
        self._solver_type = solver_type
        self._problem = None
        self.left_matrix = None
        self.right_matrix = None
        self.beta = None
        self.r0 = None

    def minimize(self, l_cs_value, r_cs_value, beta_value, component_r0, tol=1e-8):
        if self._problem is None:
            self._construct_problem(l_cs_value, r_cs_value, beta_value, component_r0)
        else:
            self._update_parameters(l_cs_value, r_cs_value, beta_value, component_r0)
        self._problem.solve(solver=self._solver_type)
        # self._problem.solve(solver='MOSEK', mosek_params={
        #     'MSK_DPAR_INTPNT_CO_TOL_PFEAS': tol,
        #     'MSK_DPAR_INTPNT_CO_TOL_DFEAS': tol,
        #     'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': tol,
        #     'MSK_DPAR_INTPNT_CO_TOL_INFEAS': tol
        # })
        self._handle_exception(self._problem)
        return self._result()

    @abstractmethod
    def _define_variables_and_parameters(self):
        pass

    def update_weights(self, weights):
        self._weights.value = weights

    def _construct_problem(self, l_cs_value, r_cs_value, beta_value, component_r0):
        self._define_variables_and_parameters(l_cs_value, r_cs_value, beta_value, component_r0)
        objective = cvx.Minimize(self._term_f1(self.left_matrix, self.right_matrix)
                                 + self._term_f2(self.left_matrix, self.right_matrix)
                                 + self._term_f3(self.left_matrix, self.right_matrix))
        constraints = self._constraints(self.left_matrix, self.right_matrix, self.beta,
                                        self.r0)
        problem = cvx.Problem(objective, constraints)
        self._problem = problem

    @abstractmethod
    def _update_parameters(self):
        pass

    def _term_f1(self, l_cs_param, r_cs_param):
        """
        This method defines the generic from of the first term of objective
        function, which calculates a quantile regression cost function,
        element-wise, between the PV power matrix (`self._power_signals_d`)
        and the low-rank model (`l_cs_param * r_cs_param`).

        Subclass defines which of l_cs and r_cs value is fixed.
        """

        weights_w1 = cvx.diag(self._weights)
        return cvx.sum((0.5 * cvx.abs(self._power_signals_d
                        - l_cs_param @ r_cs_param)
                      + (self._tau - 0.5) * (self._power_signals_d
                        - l_cs_param @ r_cs_param))
                     @ weights_w1)

    @abstractmethod
    def _term_f2(self, l_cs_param, r_cs_param):
        pass

    @abstractmethod
    def _term_f3(self, l_cs_param, r_cs_param):
        pass

    @abstractmethod
    def _constraints(self, l_cs_param, r_cs_param, beta_param, component_r0):
        pass

    @abstractmethod
    def _handle_exception(self, problem):
        pass

    def _result(self):
        return self.left_matrix.value, self.right_matrix.value, self.beta.value
