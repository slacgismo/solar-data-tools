import numpy as np
""" Longitude Direct Calculation Module
This module contains the function for the direct calculation of system
longitude based on estimated local solar noon and timezone offset from UTC.
The same exact equation is used for "Hadghdadi" and "Duffie" approaches. See
`pvsystemprofiler.utilities.equation_of_time` for equation of time (EoT)
calculations.
"""


def calculate_longitude(eot, solarnoon, days, gmt_offset):
    sn = 60 * solarnoon[days]  # convert hours to minutes
    eot_days = eot[days]
    estimates = calc_lon(sn, eot_days, gmt_offset)
    return np.nanmedian(estimates)


def calc_lon(solar_noon, eot, gmt_offset):
    """
    Using the definition of solar time and standard time, given in equation
    (1.5.2) in [1]. This is the standard equation governing the relationship
    between apparent time and the equation of time. Note that 720 is the
    number of minutes in twelve hours, so it corresponds with "standard noon"
    or 12:00PM, given in minutes since midnight.

    [1] Duffie, John A., and William A. Beckman. Solar engineering of thermal
        processes. New York: Wiley, 1991.

    :param solar_noon: time of solar noon in minutes on a given day
    :param eot: equation of time on a given day
    :param gmt_offset: local timezone offset in hours from UTC/GMT
    :return: the longitude for the given values
    """
    sn = solar_noon
    tc = 720 - sn
    lon = (tc - eot) / 4 + 15 * gmt_offset
    return lon
