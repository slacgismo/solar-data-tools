import numpy as np

def interpolate(tnew, t, x, alignment='left'):
    """
    This function interpolates a signal for a new set of points while maintaining constant signal energy.

    :param tnew: ndarray, 1D array with the new time points for interpolation (length n). Last point is the end of the last time bin.
    :param t: ndarray, 1D array with the original time points (length m).
    :param x: ndarray, 1D array with the original signal values (length m).
    :param alignment: str, time bin label alignement, either 'left' or 'right'.
    :return: ndarray, 1D array with the interpolated signal for the new time points (length n-1).
    """
    check_sanity(tnew, t, x, alignment)
    # make sure everything is float, flip if alignment is right
    tnew_float, t_float, x_float = initialize_arrays(tnew, t, x, alignment)
    # find the indices where each element in tnew would be inserted into t to maintain order
    indices = np.searchsorted(t_float, tnew_float, side='right')
    # interpolate signal values assuming a step-like function
    xnew = x_float[indices - 1]
    ttnew = np.insert(t_float, indices, tnew_float)
    xxnew = np.insert(x_float, indices, xnew)
    # indices of the new time points in the temporary array
    new_indices = indices + np.arange(tnew.shape[0])
    # get the new signal values by computing the integral
    y = compute_integral_interpolation(ttnew, xxnew, new_indices)
    # divide by the time step to maintain constant integral (energy)
    dts = np.diff(ttnew[new_indices])
    y = y / dts 
    if alignment == 'right':
        y = np.flip(y)
    return y

def check_sanity(tnew, t, x, alignment):
    """
    This functions performs basic checks on the input parameters of the interpolate function.

    :param tnew: ndarray, new time points for interpolation (length n). Last point is the end of the last time bin.
    :param t: ndarray, original time points (length m).
    :param x: ndarray, original signal values (length m).
    :param alignment: str, time bin label alignement, either 'left' or 'right'.
    :raises ValueError: raised if t and tnew do not have at least two values.
    :raises ValueError: raised if t values are not sorted.
    :raises ValueError: raised if tnew values are not sorted.
    :raises ValueError: raised if tnew values are not within the range of t values.
    :raises ValueError: raised if t does not have the same shape as x.
    :raises ValueError: raised if alignment is not 'left' or 'right'.
    """
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
    """
    This function computes the interpolation of the signal for the new time points by computing the integral.

    :param ttnew: ndarray, 1D array with the original and new time points (length m+n).
    :param xxnew: ndarray, 1D array with the original and temporary new signal values (length m+n).
    :param new_indices: ndarray, 1D array with the indices of the points in tnew (length n).
    :return: ndarray, 1D array with the interpolated signal for the new time points (length n-1).
    """
    # replace NaNs with zeros
    xxnew_filled = np.nan_to_num(xxnew, nan=0)
    # compute the piecewise than cumulative integral of the signal
    piecewise_integrals = np.diff(ttnew) * xxnew_filled[:-1]
    cumulative_integrals = np.zeros(ttnew.shape[0])
    # Add the initial zero as a baseline for the cumulative sum
    cumulative_integrals[1:] = np.cumsum(piecewise_integrals)
    # set NaNs back to Na
    was_nan = np.isnan(xxnew)
    # the last point is not used in the interpolation, it shouldn't propagate NaNs
    was_nan[-1] = False 
    cumulative_integrals[was_nan] = np.nan
    # the new value at each new time point is the difference between the cumulative integrals
    # at this point and the following one
    y = np.diff(cumulative_integrals[new_indices])
    return y

