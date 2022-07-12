import numpy as np
import cvxpy as cvx

"""
Calculates the angle incidence for a system based on its power matrix using signal decomposition.
"""


def find_fit_costheta(data_matrix, clear_index):
    """
    :param data_matrix: power matrix.
    :param clear_index: boolean array specifying clear days.
    :return: angle of incidence array.
    """
    data = np.max(data_matrix, axis=0)
    s1 = cvx.Variable(len(data))
    s2 = cvx.Variable(len(data))
    cost = 1e1 * cvx.norm(cvx.diff(s1, k=2), p=2) + cvx.norm(s2[clear_index])
    objective = cvx.Minimize(cost)
    constraints = [
        data == s1 + s2,
        s1[365:] == s1[:-365]
    ]
    problem = cvx.Problem(objective, constraints)
    problem.solve(solver='MOSEK')
    scale_factor_costheta = s1.value
    costheta_fit = data_matrix / np.max(s1.value)
    return scale_factor_costheta, costheta_fit
