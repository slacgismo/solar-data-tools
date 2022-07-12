from scipy.optimize import curve_fit
import numpy as np

""" Angle of incidence Module
This module contains the function for the calculation of the system's latitude, tilt and azimuth when some of them are
given as precalculates and others are left as unknowns. Unknowns are calculated via fit to the angle of incidence 
cos(theta) equation (1.6.2) in:
Duffie, John A., and William A. Beckman. Solar engineering of thermal processes. New York: Wiley, 1991.
"""


def run_curve_fit(func, keys,  delta, omega, costheta, boolean_filter, init_values, fit_bounds):
    """
    :param func: Angle of incidence model function.
    :param keys: Dynamic keys of parameters being calculated as returned by determine_unknowns.
    :param delta: System's declination in Degrees (array).
    :param omega: System's hour angle in Degrees(array).
    :param costheta: The dependent data. Angle of incidence array used to fit parameters.
    :param boolean_filter: boolean array specifying days to be used in fitting.
    :param init_values: Initial guess for the parameters. (Degrees).
    :param fit_bounds: Lower and upper bounds on parameters.
    :return: Optimal values for the parameters.
    """
    costheta_fit = costheta[boolean_filter]
    x = np.array([np.deg2rad(delta), np.deg2rad(omega)])

    popt, pcov = curve_fit(func, x, costheta_fit, p0=np.deg2rad(init_values), bounds=fit_bounds)

    if 'azimuth_estimate' in keys:
        popt[-1] -= np.rint(popt[-1] / 2 / np.pi) * 2 * np.pi

    estimates = np.degrees(popt)
    return estimates
