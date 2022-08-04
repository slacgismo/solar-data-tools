""" Longitude Direct Calculation Module
This module contains the function for the direct calculation of a system's
latitude based on daylight hours and the declination angle.
"""
import numpy as np


def calculate_latitude(hours_daylight, delta):
    estimates = calc_lat(hours_daylight, delta)
    return np.nanmedian(estimates)


def calc_lat(hours_daylight, delta):
    """
    Latitude is estimated from equation (1.6.11) in:
    Duffie, John A., and William A. Beckman. Solar engineering of thermal
    processes. New York: Wiley, 1991.

    :param hours_daylight: daylight hours as calculated by calculate_hours_daylight or calculate_hours_daylight_raw
    :param delta: declination as calculated from declination_equations in Degrees.
    :return: the latitude for the given values
    """
    lat = np.degrees(np.arctan(- np.cos(np.radians(15 / 2 * hours_daylight)) / (np.tan(np.deg2rad(delta[0])))))
    return lat
