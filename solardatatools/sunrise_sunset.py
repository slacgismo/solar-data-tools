''' Sunrise and Sunset Estimation Module

This module contains functions for estimating sunrise and sunset times from an
unlabel PV power dataset.
'''

import numpy as np
from solardatatools.signal_decompositions import tl1_l2d2p365

def rise_set_rough(bool_msk):
    nvals = bool_msk.shape[0]
    num_meas_per_hour = nvals / 24
    hour_of_day = np.arange(0, 24, 1. / num_meas_per_hour)
    sunrise_idxs = np.argmax(bool_msk, axis=0)
    sunset_idxs = nvals - np.argmax(np.flip(bool_msk, axis=0), axis=0) - 1
    sunrises = np.nan * np.ones_like(sunrise_idxs)
    sunsets = np.nan * np.ones_like(sunset_idxs)
    sunrises[sunrise_idxs != 0] = hour_of_day[sunrise_idxs][sunrise_idxs != 0]
    sunsets[sunset_idxs != nvals - 1] = hour_of_day[sunset_idxs][sunset_idxs != nvals - 1]
    # sunrises[np.isnan(sunrises)] = 1000
    # sunrises[sunrises > 12] = np.nan
    # sunsets[np.isnan(sunsets)] = -1000
    # sunsets[sunsets < 12] = np.nan
    return {'sunrises': sunrises, 'sunsets': sunsets}


def rise_set_smoothed(rough_dict, sunrise_tau=0.05, sunset_tau=0.95,
                      solver=None):
    sunrises = rough_dict['sunrises']
    sunsets = rough_dict['sunsets']
    sr_smoothed = tl1_l2d2p365(
        sunrises, ~np.isnan(sunrises), tau=sunrise_tau, solver=solver
    )
    ss_smoothed = tl1_l2d2p365(
        sunsets, ~np.isnan(sunsets), tau=sunset_tau, solver=solver
    )
    return {'sunrises': sr_smoothed, 'sunsets': ss_smoothed}