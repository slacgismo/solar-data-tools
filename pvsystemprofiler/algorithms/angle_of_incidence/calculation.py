import numpy as np

"""
Calculates the angle incidence using the cos(theta) equation (1.6.2) in:
Duffie, John A., and William A. Beckman. Solar engineering of thermal processes. New York: Wiley, 1991.
"""


def calculate_costheta(func, delta, omega, lat=None, tilt=None, azim=None):
    """
    :param func: angle of incidence model function.
    :param delta: System's declination in Degrees (array).
    :param omega: System's hour angle in Degrees (array).
    :param lat: System's latitude in Degrees.
    :param tilt: System's tilt in Degrees.
    :param azim: System's azimuth in Degrees.
    :return: angle of incidence array
    """
    x = np.array([np.deg2rad(delta), np.deg2rad(omega)])
    phi = np.deg2rad(lat)
    beta = np.deg2rad(tilt)
    gamma = np.deg2rad(azim)
    costheta = func(x, phi, beta, gamma)
    return costheta
