"""
This module defines functionality unique to right matrix minimization.
"""
import cvxpy as cvx
from statistical_clear_sky.algorithm.minimization.abstract\
 import AbstractMinimization
from statistical_clear_sky.algorithm.exception import ProblemStatusError

class RightMatrixModifiedMinimization(AbstractMinimization):
    """
    Uses minimization method in parent class with fixed Left Matrix value,
    keeping Right matrix as a variable.
    """

    def __init__(self, power_signals_d, rank_k, weights, tau, mu_r,
                 is_degradation_calculated=True,
                 max_degradation=0., min_degradation=-0.25,
                 non_neg_constraints=True, solver_type='ECOS'):

        super().__init__(power_signals_d, rank_k, weights, tau,
                         non_neg_constraints=non_neg_constraints, solver_type=solver_type)
        self._mu_r = mu_r

        self._is_degradation_calculated = is_degradation_calculated
        self._max_degradation = max_degradation
        self._min_degradation = min_degradation

    def _define_variables_and_parameters(self, l_cs_value, r_cs_value, beta_value, component_r0):
        self.left_matrix = cvx.Parameter(shape=(self._power_signals_d.shape[0],
                                          self._rank_k))
        self.left_matrix.value = l_cs_value
        self.right_matrix = cvx.Variable(shape=(self._rank_k,
                                         self._power_signals_d.shape[1]))
        self.right_matrix.value = r_cs_value
        self.beta = cvx.Variable()
        self.beta.value = beta_value
        self.r0 = cvx.Parameter(len(component_r0))
        self.r0.value = 1. / component_r0
        return

    def _update_parameters(self, l_cs_value, r_cs_value, beta_value, component_r0):
        self.left_matrix.value = l_cs_value
        self.beta.value = beta_value
        self.r0.value = 1. / component_r0

    def _term_f2(self, l_cs_param, r_cs_param):
        '''
        Apply smoothness constraint to all rows of right matrix
        '''
        r_tilde = self._obtain_r_tilde(r_cs_param)
        term_f2 = self._mu_r * cvx.norm(r_tilde[:, :-2] - 2
                   * r_tilde[:, 1:-1] + r_tilde[:, 2:], 'fro')
        return term_f2

    def _term_f3(self, l_cs_param, r_cs_param, beta_param, component_r0):
        '''
        Apply periodicity penalty to all rows of right matrix except the
        first one
        '''
        r_tilde = self._obtain_r_tilde(r_cs_param)
        if self._power_signals_d.shape[1] > 365:
            term_f3 = self._mu_r * cvx.norm(r_tilde[1:, :-365]
                                      - r_tilde[1:, 365:], 'fro')
        else:
            term_f3 = self._mu_r * cvx.norm(r_tilde[:, :-365]
                                      - r_tilde[:, 365:], 'fro')
        if self._power_signals_d.shape[1] > 365:
            r = r_cs_param[0, :].T
            if self._is_degradation_calculated:
                term_f3 += cvx.norm1(
                    cvx.multiply(component_r0[:-365], r[365:] - r[:-365])
                    - beta_param
                )
            else:
                term_f3 += cvx.norm1(
                    cvx.multiply(component_r0[:-365], r[365:] - r[:-365])
                )
        return term_f3

    def _constraints(self, l_cs_param, r_cs_param, beta_param, component_r0):
        constraints = []
        # if self._power_signals_d.shape[1] > 365:
        #     r = r_cs_param[0, :].T
        #     if self._is_degradation_calculated:
        #         constraints.extend([
        #             cvx.multiply(component_r0[:-365], r[365:] - r[:-365]) == beta_param
        #         ])
        #         if self._max_degradation is not None:
        #             constraints.append(
        #                 beta_param <= self._max_degradation)
        #         if self._min_degradation is not None:
        #             constraints.append(
        #                 beta_param >= self._min_degradation)
        #     else:
        #         constraints.append(cvx.multiply(component_r0[:-365],
        #                                         r[365:] - r[:-365]) == 0)
        if self._non_neg_constraints:
            constraints.extend([
                l_cs_param @ r_cs_param >= 0,
                r_cs_param[0] >= 0
            ])
        return constraints

    def _handle_exception(self, problem):
        if problem.status != 'optimal':
            raise ProblemStatusError('Minimize R status: ' + problem.status)

    def _obtain_r_tilde(self, r_cs_param):
        '''
        This function handles the smoothness and periodicity constraints when
        the data set is less than a year long. It operates by filling out the
        rest of the year with blank variables, which are subsequently dropped
        after the problem is solved.

        :param r_cs_param: the right matrix CVX variable
        :return: A cvx variable with second dimension at least 367
        '''
        if r_cs_param.shape[1] < 365 + 2:
            n_tilde = 365 + 2 - r_cs_param.shape[1]
            r_tilde = cvx.hstack([r_cs_param,
                                  cvx.Variable(shape=(self._rank_k, n_tilde))])
        else:
            r_tilde = r_cs_param
        return r_tilde
