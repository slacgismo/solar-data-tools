"""
This module defines functionality unique to left matrix minimization.
"""
import cvxpy as cvx
import numpy as np
from statistical_clear_sky.algorithm.minimization.abstract\
 import AbstractMinimization
from statistical_clear_sky.algorithm.exception import ProblemStatusError

class LeftMatrixMinimization(AbstractMinimization):
    """
    Uses minimization method in parent class with fixed Right matrix value,
    keeping Left matrix as a variable.
    """

    def __init__(self, power_signals_d, rank_k, weights, tau, mu_l,
                 non_neg_constraints=True, solver_type='ECOS'):

        super().__init__(power_signals_d, rank_k, weights, tau,
                         non_neg_constraints=non_neg_constraints, solver_type=solver_type)
        self._mu_l = mu_l

    def _define_variables_and_parameters(self, l_cs_value, r_cs_value, beta_value, component_r0):
        self.left_matrix = cvx.Variable(shape=(self._power_signals_d.shape[0],
                                         self._rank_k))
        self.left_matrix.value = l_cs_value
        self.right_matrix = cvx.Parameter(shape=(self._rank_k,
                                   self._power_signals_d.shape[1]))
        self.right_matrix.value = r_cs_value
        self.beta = cvx.Variable()
        self.beta.value = beta_value
        self.r0 = cvx.Parameter(len(component_r0))
        self.r0.value = 1. / component_r0
        return

    def _update_parameters(self, l_cs_value, r_cs_value, beta_value, component_r0):
        self.right_matrix.value = r_cs_value
        self.beta.value = beta_value
        self.r0.value = 1. / component_r0

    def _term_f2(self, l_cs_param, r_cs_param):
        weights_w2 = np.eye(self._rank_k)
        term_f2 = self._mu_l * cvx.norm((l_cs_param[:-2, :] - 2
                * l_cs_param[1:-1, :] + l_cs_param[2:, :]) @ weights_w2, 'fro')
        return term_f2

    def _term_f3(self, l_cs_param, r_cs_param):
        return 0

    def _constraints(self, l_cs_param, r_cs_param, beta_param, component_r0):
        constraints = [cvx.sum(l_cs_param[:, 1:], axis=0) == 0]
        ixs = self._handle_bad_night_data()
        if sum(ixs) > 0:
            constraints.append(l_cs_param[ixs, :] == 0)
        if self._non_neg_constraints:
            constraints.extend([
                l_cs_param @ r_cs_param >= 0,
                l_cs_param[:, 0] >= 0
            ])
        return constraints

    def _handle_exception(self, problem):
        if problem.status != 'optimal':
            raise ProblemStatusError('Minimize L status: ' + problem.status)

    def _handle_bad_night_data(self):
        '''
        Method for generating the "nighttime" index set

        This method finds the (approximate) set of time stamps that correspond with
        nighttime across all seasons in the given data.

        Old method looked for timestamps with an average power across all days that is
        smaller than 0.5% of the max power value in the data set.

        New method still looks for timestamps with values below 0.5% of the max, but
        then converts this to a sparsity by row, and returns the rows with a sparsity
        of greater than 96%. This approach is more robust than the old method because
        it is not sensitive to the magnitude of any spurious nighttime data values.
        :return:
        '''
        data = self._power_signals_d
        row_sparsity = 1 - np.sum(data > 0.005 * np.max(data), axis = 1) / data.shape[1]
        threshold = 0.96
        #ix_array = np.average(self._power_signals_d, axis=1) / np.max(
        #    np.average(self._power_signals_d, axis=1)) <= 0.005
        ix_array = row_sparsity >= threshold
        return ix_array
