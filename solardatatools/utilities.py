import numpy as np
import cvxpy as cvx

def total_variation_filter(signal, C=5):
    '''
    This function performs total variation filtering or denoising on a 1D signal. This filter is implemented as a
    convex optimization problem which is solved with cvxpy.
    (https://en.wikipedia.org/wiki/Total_variation_denoising)

    :param signal: A 1d numpy array (must support boolean indexing) containing the signal of interest
    :param C: The regularization parameter to control the total variation in the final output signal
    :return: A 1d numpy array containing the filtered signal
    '''
    s_hat = cvx.Variable(len(signal))
    mu = cvx.Constant(value=C)
    index_set = ~np.isnan(signal)
    objective = cvx.Minimize(cvx.sum(cvx.huber(signal[index_set] - s_hat[index_set]))
                             + mu * cvx.norm1(cvx.diff(s_hat, k=1)))
    problem = cvx.Problem(objective=objective)
    try:
        problem.solve(solver='MOSEK')
    except Exception as e:
        print(e)
        print('Trying ECOS solver')
        problem.solve(solver='ECOS')
    return s_hat.value

def total_variation_plus_seasonal_filter(signal, c1=10, c2=500):
    '''
    This performs total variation filtering with the addition of a seasonal baseline fit. This introduces a new
    signal to the model that is smooth and periodic on a yearly time frame. This does a better job of describing real,
    multi-year solar PV power data sets, and therefore does an improved job of estimating the discretely changing
    signal.

    :param signal: A 1d numpy array (must support boolean indexing) containing the signal of interest
    :param c1: The regularization parameter to control the total variation in the final output signal
    :param c2: The regularization parameter to control the smoothness of the seasonal signal
    :return: A 1d numpy array containing the filtered signal
    '''
    s_hat = cvx.Variable(len(signal))
    s_seas = cvx.Variable(len(signal))
    s_error = cvx.Variable(len(signal))
    c1 = cvx.Constant(value=c1)
    c2 = cvx.Constant(value=c2)
    index_set = ~np.isnan(signal)
    w = len(signal) / np.sum(index_set)
    objective = cvx.Minimize(
        (365 * 3 / len(signal)) * w * cvx.sum(cvx.huber(s_error))
        + c1 * cvx.norm1(cvx.diff(s_hat, k=1))
        + c2 * cvx.norm(cvx.diff(s_seas, k=2))
        + c2 * .1 * cvx.norm(cvx.diff(s_seas, k=1))
    )
    constraints = [
        signal[index_set] == s_hat[index_set] + s_seas[index_set] + s_error[index_set],
        s_seas[365:] - s_seas[:-365] == 0,
        cvx.sum(s_seas) == 0
    ]
    problem = cvx.Problem(objective=objective, constraints=constraints)
    problem.solve()
    return s_hat.value, s_seas.value


def progress(count, total, status=''):
    """
    Python command line progress bar in less than 10 lines of code. · GitHub
    https://gist.github.com/vladignatyev/06860ec2040cb497f0f3
    :param count: the current count, int
    :param total: to total count, int
    :param status: a message to display
    :return:
    """
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
sys.stdout.flush()