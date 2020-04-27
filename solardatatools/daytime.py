''' Daytime Module

This module contains a function for finding the daytime period of a power time
series

'''

import numpy as np

def find_daytime(data_matrix):
    mat_copy = np.copy(data_matrix)
    np.nan_to_num(mat_copy, copy=False, nan=0.0)
    mat_copy -= np.quantile(mat_copy, 0.05)
    mat_copy /= np.max(mat_copy)
    daytime_mask = mat_copy >= 0.01
    return daytime_mask
