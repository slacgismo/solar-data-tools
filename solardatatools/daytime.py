''' Daytime Module

This module contains a function for finding the daytime period of a power time
series

'''

import numpy as np

def find_daytime(data_matrix, threshold=0.01):
    """
    Function for creating a boolean mask of time periods when the sun is
    up on a PV system, based on a a power signal. The data is scaled so that
    "night time" data is approximately zero and the maximum power output
    approximately 1. We take the 5th and 95th percentiles for the top and
    bottom scale factors instead of using the maximum and the minimum. This
    provides robustness again outlier measurements. Then, daytime is chosen
    as scaled values larger than the threshold, which has a default value of
    0.01, i.e. power measurements larger than 1% of the 95th percentile. If
    the 95th percentile corresponds to "one sun" of irradiance or 1000 W/m^2,
    this default threshold value represents irradiance levels above 10 W/m^2.

    :param data_matrix: A power data matrix, possibly with NaN values
    :param threshold: daytime threshold
    :return: boolean mask of daytime periods, matching the power matrix
    """
    mat_copy = np.copy(data_matrix)
    # convert NaNs to zeros
    np.nan_to_num(mat_copy, copy=False, nan=0.0)
    # scale mat_copy to mostly have values >=0 and <= 1, allowing for outliers
    bottom_scale = max(np.quantile(mat_copy, 0.05), 0)
    top_scale = np.quantile(mat_copy, 0.95)
    mat_copy -= bottom_scale
    mat_copy /= top_scale - bottom_scale
    # define boolean mask rule
    daytime_mask = mat_copy >= threshold
    return daytime_mask

def detect_sun(data, threshold):
    scaled_mat = scale_data(data)
    bool_msk = np.zeros_like(scaled_mat, dtype=np.bool)
    slct = ~np.isnan(scaled_mat)
    bool_msk[slct] = scaled_mat[slct] > threshold
    return bool_msk

def scale_data(data, return_metrics=False):

    high_val = np.nanquantile(data, .99)
    low_val = max(np.nanmin(data), -0.005 * high_val)
    scaled_mat = (data - low_val) / high_val
    nan_mask = np.isnan(scaled_mat)
    if not return_metrics:
        return scaled_mat
    else:
        m1 = low_val
        m2 = np.sum(nan_mask) / data.size
        m3 = np.nanquantile(scaled_mat, 0.5) / np.nanquantile(scaled_mat, 0.99)
        return scaled_mat, m1, m2, m3