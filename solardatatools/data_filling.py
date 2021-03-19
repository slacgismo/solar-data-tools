# -*- coding: utf-8 -*-
''' Data Filling Module

This module contains functions for filling missing data in a PV power matrix

'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from solardatatools.daytime import find_daytime
from solardatatools.algorithms import SunriseSunset

def zero_nighttime(data_matrix, night_mask=None, daytime_threshold=0.005):
    D = np.copy(data_matrix)
    D[D < 0] = 0
    if night_mask is None:
        ss = SunriseSunset()
        ss.calculate_times(data_matrix, threshold=daytime_threshold)
        night_mask = ~ss.sunup_mask_estimated
    D[np.logical_and(night_mask, np.isnan(D))] = 0
    return D

def interp_missing(data_matrix):
    D = np.copy(data_matrix)
    D_df = pd.DataFrame(data=D)
    D = D_df.interpolate().values
    return D
