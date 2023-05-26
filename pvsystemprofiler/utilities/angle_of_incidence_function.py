"""The function cos(theta) is  calculated using equation (1.6.2) in:
 Duffie, John A., and William A. Beckman. Solar engineering of thermal
 processes. New York: Wiley, 1991."""

import numpy as np
from pvsystemprofiler.utilities.hour_angle_equation import calculate_omega
from pvsystemprofiler.utilities.declination_equation import delta_cooper


def func_costheta(x, phi, beta, gamma):
    delta = x[0]
    omega = x[1]

    gamma -= np.rint(gamma / 2 / np.pi) * 2 * np.pi

    a = np.sin(delta) * np.sin(phi) * np.cos(beta)
    b = np.sin(delta) * np.cos(phi) * np.sin(beta) * np.cos(gamma)
    c = np.cos(delta) * np.cos(phi) * np.cos(beta) * np.cos(omega)
    d = np.cos(delta) * np.sin(phi) * np.sin(beta) * np.cos(gamma) * np.cos(omega)
    e = np.cos(delta) * np.sin(beta) * np.sin(gamma) * np.sin(omega)
    return a - b + c + d + e


def costheta_calc_helper(
    tilt, azimuth, latitude, longitude, tz_offset, data_handler=None, data_sampling=1
):
    """Returns a numpy array with the same shape as the data matrix
    attribute on the input data handler. Each entry contains the the calculate
    cosine of the solar angle of incidence for that timestamp.

    :param data_handler: an SDT DataHandler instance, loaded with a data set,
        and having run the standard pipeline
    :type data_handler: solardatatools.DataHandler class intstance
    :param tilt: the angle between the plane of the array and the horizontal
        in degrees 0 <= tilt <= 180 (tilt > 90 implies downward component)
    :type tilt: float
    :param azimuth: the deviation of the projection on a horizonal plane of the
        normal to the plane of the array to the surface from the local
        meridian in degrees, with 0 being north and 0 <= azimuth <= 360, going
        N -> E -> S -> W
    :type azimuth: float
    :param latitude: the angular location north or south of the equator, north
        positive, in degrees -90 <= latitude <= 90
    :type latitude: float
    :param longitude: the angular location east or west of the prime meridian,
        with east positive and west negative, in degrees
        -180 <= longitude <= 180
    :type longitude: float
    :param tz_offset: the UTC offset associated with the meridian defining the
        local standard time (e.g. Pacific standard time is -8)
    :type tz_offset: integer
    """
    if data_handler is not None:
        dh = data_handler
        doy = dh.day_index.day_of_year
        meas_per_day = dh.filled_data_matrix.shape[0]
        data_sampling = dh.data_sampling
    else:
        doy = np.arange(365) + 1
        meas_per_day = int(1440 / data_sampling)
        data_sampling = data_sampling
    delta = delta_cooper(doy, meas_per_day)
    omega = calculate_omega(doy, data_sampling, longitude, tz_offset)
    # func_costheta uses the convention from Duffie and Beckman that zero is
    # due south, but the industry uses the 'meteorological' convention that
    # zero is north. This converts between the conventions
    az_duff = azimuth - 180
    costheta = func_costheta(
        (np.deg2rad(delta), np.deg2rad(omega)),
        np.deg2rad(latitude),
        np.deg2rad(tilt),
        np.deg2rad(az_duff),
    )
    return costheta
