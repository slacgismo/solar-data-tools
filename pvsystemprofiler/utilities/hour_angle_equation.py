"""Omega, the hour angle is estimated as defined on p. 13 in:
       Duffie, John A., and William A. Beckman. Solar engineering of thermal
       processes. New York: Wiley, 1991."""

import numpy as np
from pvsystemprofiler.utilities.time_convert import clock_to_solar


def calculate_omega(doy, data_sampling, lon, gmt_offset):
    """
    :param data_sampling: daily data sampling.
    :param num_days: number of sampling days.
    :param lon: longitude in Degrees (float)
    :param doy: day of year (float or array)
    :param gmt_offset: local timezone offset in hours from UTC/GMT (float or int)
    :return: hour angle omega (float or array)
    """
    num_days = len(doy)
    minutes_day = np.arange(0, 1440, data_sampling)
    minutes_doy = np.tile(minutes_day.reshape(-1, 1), (1, num_days))
    hours_doy_solar = clock_to_solar(minutes_doy, lon, doy, gmt_offset, eot="duffie")
    hours_doy_solar /= 60
    omega = 15 * (hours_doy_solar - 12)
    return omega
