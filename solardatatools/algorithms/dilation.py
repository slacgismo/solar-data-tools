""" Dilation Module

This module contains functions to dilate a signal from a regular time grid to a dilated time grid and
to undilate it back.

"""

import numpy as np
from solardatatools.plotting import plot_2d

DEFAULT = {
    "nvals_dil": 101,
}


class Dilation:
    def __init__(self, data_handler, **config):
        self.dh = data_handler
        self.nvals_ori = data_handler.raw_data_matrix.shape[0]
        self.ndays = data_handler.raw_data_matrix.shape[1]
        self.idx_ori = None
        self.idx_dil = None
        self.signal_ori = data_handler.raw_data_matrix.ravel(order="F")
        self.signal_dil = None
        if len(config) == 0:
            self.config = DEFAULT
        else:
            self.config = config
        self.nvals_dil = self.config["nvals_dil"]
        self.run()

    def run(self):
        # original index as 1D array of floats (hours since midnight of the first day)
        self.idx_ori = np.linspace(0, self.ndays * 24, self.ndays * self.nvals_ori + 1)
        # dilated index as 1D float array of floats (hours since midnight of the first day)
        self.idx_dil = build_dilated_idx(
            self.dh.daytime_analysis.sunrise_estimates,
            self.dh.daytime_analysis.sunset_estimates,
            self.idx_ori,
            self.config["nvals_dil"],
        )
        # dilated signal
        self.signal_dil = dilate_signal(self.idx_dil, self.idx_ori, self.signal_ori)

    def plot_heatmap(
        self,
        space="original",
        figsize=(12, 6),
        scale_to_kw=True,
        year_lines=True,
        units=None,
    ):
        if space == "original":
            mat = self.signal_ori.reshape((self.nvals_ori, self.ndays), order="F")
        elif space == "dilated":
            mat = self.signal_dil[1:].reshape(
                (self.config["nvals_dil"], self.ndays), order="F"
            )
        else:
            raise ValueError(
                "Invalid value for space. Choose from: ['original', 'dilated']"
            )
        if units is None:
            if scale_to_kw and self.dh.power_units == "W":
                mat /= 1000
                units = "kW"
            else:
                units = self.dh.power_units
        return plot_2d(
            mat,
            figsize=figsize,
            dates=self.dh.day_index,
            year_lines=year_lines,
            units=units,
        )


def build_dilated_idx(
    sunrises, sunsets, signal_idx_ori, nvals_dil=DEFAULT["nvals_dil"]
):
    """
    This function builds a float index from 00:00 first day to the last sunset of the last day (eg. 18:37)
    for the dilated signal. The last point of the index is the end of the last time bin (00:00 next day).

    :param sunrises: ndarray, 1D array with the sunrise times (length ndays).
    :param sunsets: ndarray, 1D array with the sunset times (length ndays).
    :param signal_idx_ori: ndarray, 1D array with the original index (length nvals_ori*ndays + 1).
    :param nvals_dil: int, number of values per day for the dilated signal. Default is 101.
    :return: ndarray, 1D array with the dilated index (length nvals_dil*ndays + 2).
    """
    sunrise_idx_ori = sunrises + 24 * np.arange(len(sunrises))
    sunset_idx_ori = sunsets + 24 * np.arange(len(sunsets))
    signal_idx_dil = np.linspace(sunrise_idx_ori, sunset_idx_ori, nvals_dil).ravel(
        order="F"
    )
    signal_idx_dil = np.append(0, signal_idx_dil)  # Adding first midnight
    signal_idx_dil = np.append(signal_idx_dil, signal_idx_ori[-1])  # Last bin end
    return signal_idx_dil


def dilate_signal(signal_idx_dil, signal_idx_ori, signal_ori):
    """
    This function dilates a signal from a regular time grid to a dilated time grid.

    :param signal_idx_dil: ndarray, 1D array with the dilated index (length n).
    :param signal_idx_ori: ndarray, 1D array with the original index (length m).
    :param signal_ori: ndarray, 1D array with the original signal values (length m-1).
    :return: ndarray, 1D array with the dilated signal values (length n-1).
    """
    _signal_ori = np.append(
        signal_ori, signal_ori[-1]
    )  # Adding last dummy value to interpolate
    signal_dil = interpolate(
        signal_idx_dil, signal_idx_ori, _signal_ori, alignment="left"
    )
    return signal_dil


def undilate_signal(signal_idx_ori, signal_idx_dil, signal_dil):
    """
    This function undilates a signal from the dilated time grid to the regular time grid.

    :param signal_idx_ori: ndarray, 1D array with the original index (length m).
    :param signal_idx_dil: ndarray, 1D array with the dilated index (length n).
    :param signal_dil: ndarray, 1D array with the dilated signal values (length n-1).
    :return: ndarray, 1D array with the original signal values (length m-1).
    """
    _signal_dil = np.append(
        signal_dil, signal_dil[-1]
    )  # Adding last dummy value to interpolate
    signal_ori = interpolate(
        signal_idx_ori, signal_idx_dil, _signal_dil, alignment="left"
    )
    return signal_ori


def undilate_quantiles(
    signal_idx_ori, signal_idx_dil, quantiles_dil, nvals_dil=DEFAULT["nvals_dil"]
):
    """
    This function undilates a 2D matrix of quantiles from the dilated time grid to the regular time grid.

    :param signal_idx_ori: ndarray, 1D array with the original index (length m).
    :param signal_idx_dil: ndarray, 1D array with the dilated index (length n).
    :param quantiles_dil: ndarray, 2D array with the quantile values in the dilated time grid (shape (n-1, p)).
    :param nvals_dil: int, number of values per day for the dilated signal. Default is 101.
    :return: ndarray, 2D array with the quantile values in the original time grid (shape (m-1, p)).
    """
    # we remove the first and last points of the dilated signal index to get the number of days
    ndays = (len(signal_idx_dil) - 2) // nvals_dil
    # we add one zero value at the beginning of every every night
    new_signal_idx_dil = extrapolate_signal_after_sunset(
        signal_idx_dil, nvals_dil, ndays, method="linear"
    )
    _quantile_dil = np.zeros(nvals_dil * ndays + 2)
    quantiles_ori = np.zeros((signal_idx_ori.shape[0] - 1, quantiles_dil.shape[1]))
    for i in range(quantiles_dil.shape[1]):
        _quantile_dil[:-1] = quantiles_dil[:, i]
        new_quantile_dil = extrapolate_signal_after_sunset(
            _quantile_dil, nvals_dil, ndays, method="zero_padding"
        )
        quantiles_ori[:, i] = interpolate(
            signal_idx_ori, new_signal_idx_dil, new_quantile_dil, alignment="left"
        )
    return quantiles_ori


def extrapolate_signal_after_sunset(signal, nvals, ndays, method):
    """
    This generic function extrapolates a signal after sunset to add a zero value
    at the beginning of every night.

    :param signal: ndarray, 1D array with the signal values (length nvals*ndays + 2).
    :param nvals: int, number of values per day for the signal.
    :param ndays: int, number of days in the signal.
    :param method: str, method to use for the extrapolation, either 'linear' or 'zero_padding'.
    :raises ValueError: raised if method is not 'linear' or 'zero_padding'.
    :return: ndarray, 1D array with the signal values after the extrapolation (length (nvals+1)*ndays + 2).
    """
    # Signal has length nvals * ndays + 2
    matrix = np.zeros((nvals + 1, ndays))
    matrix[:-1] = signal[1:-1].reshape((nvals, ndays), order="F")
    if method == "linear":
        matrix[-1] = matrix[-2] + (matrix[-2] - matrix[-3])
    elif method == "zero_padding":
        matrix[-1] = 0
    else:
        raise ValueError(
            "Invalid value for method. Choose from: ['linear', 'zero_padding']"
        )
    new_signal = np.zeros((nvals + 1) * ndays + 2)
    new_signal[0] = signal[0]
    new_signal[1:-1] = matrix.ravel(order="F")
    new_signal[-1] = signal[-1]
    return new_signal


def interpolate(tnew, t, x, alignment="left"):
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
    indices = np.searchsorted(t_float, tnew_float, side="right")
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
    if alignment == "right":
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
    if alignment not in ["left", "right"]:
        raise ValueError("Invalid value for alignment. Choose from: ['left', 'right']")


def initialize_arrays(tnew, t, x, alignment):
    if alignment == "left":
        tnew_float = tnew.astype(float)
        t_float = t.astype(float)
        x_float = x.astype(float)
    if alignment == "right":
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
