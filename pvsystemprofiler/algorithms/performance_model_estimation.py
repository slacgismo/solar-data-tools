import numpy as np
import cvxpy as cvx

"""
Calculates the angle incidence for a system based on its power matrix using signal decomposition.
"""


def find_fit_costheta(data_matrix, clear_index, doy):
    """
    Fits a 'cos(theta)' curve to the given power, which is assumed to be smooth, yearly periodic, and symmetric around
    the summer solstice. Previous versions of this function did not enforce the symmetric constraint.

    :param data_matrix: power matrix.
    :param clear_index: boolean array specifying clear days.
    :param doy: day of year (float or array)
    :return: angle of incidence array.
    """
    data = np.max(data_matrix, axis=0)
    msk = clear_index
    y = np.zeros_like(data)
    y[msk] = np.log(data[msk])
    y[~msk] = np.nan

    x1 = cvx.Variable(len(data))
    x2 = cvx.Variable(len(data))
    x3 = cvx.Variable(len(data))
    x4 = cvx.Variable(len(data))

    constraints = [y[msk] == (x1 + x2 + x3 + x4)[msk]]

    phi1 = (100 / len(data)) * cvx.sum_squares(x1)
    phi2 = 0
    z = cvx.Variable()
    constraints.append(cvx.diff(x2, k=1) == z)
    phi3 = 1e3 * cvx.sum_squares(cvx.diff(x3, k=2)) + 1e-6 * cvx.norm1(x3)
    for val in set(solstice_centered_index(doy)):
        constraints.append(cvx.diff(x3[solstice_centered_index(doy) == val]) == 0)
    constraints.append(x3 <= 0)
    phi4 = 1e3 * cvx.sum_squares(cvx.diff(x4, k=2)) + 1e-3 * cvx.sum_squares(x4)
    constraints.append(x4[:-365] == x4[365:])
    constraints.append(x4 <= 0)
    objective = phi1 + phi2 + phi3 + phi4

    problem = cvx.Problem(cvx.Minimize(objective), constraints)
    problem.solve(solver="OSQP")
    normalized_data = data_matrix / np.exp(x2.value + x4.value)
    costheta_est = x3.value
    return normalized_data, costheta_est


def solstice_centered_index(doy):
    y = np.atleast_1d(doy)
    s1 = y < 172
    s2 = np.logical_and(y > 172, y <= 354)
    s3 = y > 354
    out = np.zeros_like(y)
    out[s1] = 172 - y[s1]
    out[s2] = y[s2] - 172
    out[s3] = 172 - (y[s3] - 365)
    if len(out) == 1:
        out = out[0]
    return out
