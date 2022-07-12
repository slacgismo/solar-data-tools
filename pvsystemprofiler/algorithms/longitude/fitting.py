import numpy as np
import cvxpy as cvx


def fit_longitude(eot, solarnoon, days, gmt_offset, loss='l2'):
    lon = cvx.Variable()
    if loss == 'l2':
        cost_func = cvx.norm
    elif loss == 'l1':
        cost_func = cvx.norm1
    elif loss == 'huber':
        cost_func = lambda x: cvx.sum(cvx.huber(x))

    sn_m = 720 - eot + 4 * (15 * gmt_offset - lon)
    sn_h = sn_m / 60
    nan_mask = np.isnan(solarnoon)
    use_days = np.logical_and(days, ~nan_mask)
    cost = cost_func(sn_h[use_days] - solarnoon[use_days])
    objective = cvx.Minimize(cost)
    problem = cvx.Problem(objective)
    problem.solve()
    return lon.value.item()
