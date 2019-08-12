# -*- coding: utf-8 -*-
''' Data Filling Module

This module contains functions for filling missing data in a PV power matrix

'''

import numpy as np
import pandas as pd

def zero_nighttime(data_matrix):
    D = np.copy(data_matrix)
    try:
        with np.errstate(invalid='ignore'):
            night_msk = D < 0.005 * np.max(D[~np.isnan(D)])
    except ValueError:
        night_msk = D < 0.005 * np.max(D)
    D[night_msk] = np.nan
    good_vals = (~np.isnan(D)).astype(int)
    sunrise_idxs = np.argmax(good_vals, axis=0)
    sunset_idxs = D.shape[0] - np.argmax(np.flip(good_vals, 0), axis=0)
    D_msk = np.zeros_like(D, dtype=np.bool)
    for ix in range(D.shape[1]):
        if sunrise_idxs[ix] > 0:
            D_msk[:sunrise_idxs[ix] - 1, ix] = True
            D_msk[sunset_idxs[ix] + 1:, ix] = True
        else:
            D_msk[:, ix] = True
    D[D_msk] = 0
    return D

def interp_missing(data_matrix):
    D = np.copy(data_matrix)
    D_df = pd.DataFrame(data=D)
    D = D_df.interpolate().values
    return D
