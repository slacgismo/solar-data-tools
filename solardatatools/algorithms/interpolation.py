import numpy as np

def check_sanity(tnew, t, x, alignment):
    if (t.shape[0] < 2) | (tnew.shape[0] < 2):
        raise ValueError("t and tnew must have at least two values.")
    if not np.all(np.diff(t) >= 0):
        raise ValueError("t values must be sorted.")
    if not np.all(np.diff(tnew) >= 0):
        raise ValueError("tnew values must be sorted.")
    if (tnew[0] < t[0]) | (tnew[-1] > t[-1]):
        raise ValueError("tnew values must be within the range of t values.")
    if t.shape != x.shape:
        raise ValueError("t must have the same shape as x.")
    if alignment not in ['left', 'right']:
        raise ValueError("Invalid value for alignment. Choose from: ['left', 'right']")

def initialize_arrays(tnew, t, x, alignment):
    if alignment == 'left':
        tnew_float = tnew.astype(float)
        t_float = t.astype(float)
        x_float = x.astype(float)
    if alignment == 'right':
        tnew_float = -np.flip(tnew).astype(float)
        t_float = -np.flip(t).astype(float)
        x_float = np.flip(x).astype(float)
    return tnew_float, t_float, x_float

def compute_integral_interpolation(ttnew, xxnew, new_indices):
    '''
    Parameters:
        ttnew (np.ndarray): t + tnew (length m+n).
        xxnew (np.ndarray): x + xnew (length m+n).
        new_indices (np.ndarray): Indices of the points in tnew (length n).

    Returns:
        y (np.ndarray): Interpolated signal (length n-1).
    '''
    piecewise_integrals = np.diff(ttnew) * xxnew[:-1]
    cumulative_integrals = np.zeros(ttnew.shape[0])
    # Add the initial zero as a baseline for the cumulative sum
    cumulative_integrals[1:] = np.cumsum(piecewise_integrals)
    y = np.diff(cumulative_integrals[new_indices])
    return y

def interpolate(tnew, t, x, alignment='left'):
    '''
    Interpolate a signal for a new set of points while maintaining constant signal energy.

    Parameters:
        tnew (np.ndarray): New time points for interpolation (length n).
            Last point is the end of the last time bin.
        t (np.ndarray): Original time points (length m).
        x (np.ndarray): Original signal values (length m).
        alignment (str): Time bin label alignement, either 'left' or 'right'.

    Returns:
        y (np.ndarray): Interpolated signal for the new time points (length n-1).
    '''

    check_sanity(tnew, t, x, alignment)
    # make sure everything is float, flip if alignment is right
    tnew_float, t_float, x_float = initialize_arrays(tnew, t, x, alignment)
    indices = np.searchsorted(t_float, tnew_float, side='right')
    xnew = x_float[indices - 1]
    ttnew = np.insert(t_float, indices, tnew_float)
    xxnew = np.insert(x_float, indices, xnew)
    new_indices = indices + np.arange(tnew.shape[0])
    y = compute_integral_interpolation(ttnew, xxnew, new_indices)
    dts = np.diff(ttnew[new_indices])
    y = y / dts # normalize time bin length to keep the integral constant
    if alignment == 'right':
        y = np.flip(y)
    return y