"""
This module defines helper class for the purpose of linearization.
(Named as a helper instead of util since it doesn't directly do liniearization.)
"""

import numpy as np
import cvxpy as cvx

class LinearizationHelper(object):
    """
    Delegate class to take care of obtaining a value used to make make a
    constraint to be linear, in order to make the optimization problem to
    be convex optimization problem.
    """

    def __init__(self, solver_type='ECOS'):
        """
        Keyword arguments
        -----------------
        solver_type : SolverType Enum
            Type of solver.
            See statistical_clear_sky.solver_type.SolverType for valid solvers.
        """
        self._solver_type = solver_type

    def obtain_component_r0(self, initial_r_cs_value, index_set=None):
        """
        Obtains the initial r0 values that are used in place of variables
        denominator of degradation equation.
        Removed duplicated code from the original implementation.

        Arguments
        -----------------
        initial_r_cs_value : numpy array
            Initial low dimension right matrix.

        Returns
        -------
        numpy array
            The values that is used in order to make the constraint of
            degradation to be linear.
        """

        component_r0 = initial_r_cs_value[0]
        if index_set is None:
            index_set = component_r0 > 1e-3 * np.percentile(component_r0, 95)
        x = cvx.Variable(initial_r_cs_value.shape[1])
        objective = cvx.Minimize(
            cvx.sum(0.5 * cvx.abs(component_r0[index_set] - x[index_set]) + (.9 - 0.5) *
                    (component_r0[index_set] - x[index_set])) + 1e3 * cvx.norm(cvx.diff(x, k=2)))
        if initial_r_cs_value.shape[1] > 365:
            constraints = [cvx.abs(x[365:] - x[:-365]) <= 1e-2 * np.percentile(component_r0, 95)]
        else:
            constraints = []
        problem = cvx.Problem(objective, constraints)
        problem.solve(solver=self._solver_type)
        result_component_r0 = x.value
 
        return result_component_r0
