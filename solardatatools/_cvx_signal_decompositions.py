import sys
import numpy as np

import cvxpy as cvx

def _cvx_l2_l1d1_l2d2p365(
    signal,
    use_ixs=None,
    w0=10, # "hard-coded"
    w1=50, # optimized
    w2=1e5,
    yearly_periodic=False,
    return_all=False, 
    solver="MOSEK",
    transition_locs=None,
    verbose=True,
):
    """
    This performs total variation filtering with the addition of a seasonal
    baseline fit. This introduces a new signal to the model that is smooth and
    periodic on a yearly time frame. This does a better job of describing real,
    multi-year solar PV power data sets, and therefore does an improved job of
    estimating the discretely changing signal.

    :param signal: A 1d numpy array (must support boolean indexing) containing
    the signal of interest
    :param w1: The regularization parameter to control the total variation in
    the final output signal
    :param w2: The regularization parameter to control the smoothness of the
    seasonal signal
    :return: A 1d numpy array containing the filtered signal
    """
    if use_ixs is None:
        index_set = ~np.isnan(signal)
    else:
        index_set = np.logical_and(use_ixs, ~np.isnan(signal))

    # Iterative reweighted L1 heuristic
    tv_weights = np.ones(len(signal) - 1)
    eps = 0.1
    n_iter = 5

    w0 = cvx.Constant(value=w0)
    w1 = cvx.Constant(value=w1)
    w2 = cvx.Constant(value=w2)

    for i in range(n_iter):
        s_hat = cvx.Variable(len(signal))
        s_seas = cvx.Variable(len(signal))
        s_error = cvx.Variable(len(signal))

        if transition_locs is None: # TODO: this should be two separate sd problems
            objective = cvx.Minimize(
                w0 * cvx.sum_squares(s_error)
                + w1 * cvx.norm1(cvx.multiply(tv_weights, cvx.diff(s_hat, k=1)))
                + w2 * cvx.sum_squares(cvx.diff(s_seas, k=2))
            )
        else:
            objective = cvx.Minimize(
                w0 * cvx.norm(s_error)
               + w2 * cvx.sum_squares(cvx.diff(s_seas, k=2))
            )
        # Consistency constraints
        constraints = [
            signal[index_set] == s_hat[index_set] + s_seas[index_set] + s_error[index_set],
            cvx.sum(s_seas[:365]) == 0,
        ]
        if len(signal) > 365:
            constraints.append(s_seas[365:] - s_seas[:-365] == 0)
            if yearly_periodic:
                constraints.append(s_hat[365:] - s_hat[:-365] == 0)
        if transition_locs is not None:
            loc_mask = np.ones(len(signal) - 1, dtype=bool)
            loc_mask[transition_locs] = False
            constraints.append(cvx.diff(s_hat, k=1)[loc_mask] == 0)

        problem = cvx.Problem(objective=objective, constraints=constraints)
        problem.solve(solver=solver, verbose=verbose)

        tv_weights = 1 / (eps + np.abs(np.diff(s_hat.value, n=1)))

    if return_all:
        return s_hat.value, s_seas.value, s_error.value, problem.objective.value

    return s_hat.value, s_seas.value