"""
This module is used to set the hour_angle_equation in terms of the unknowns. The hour equation is a function of the
declination (delta), the hour angle (omega) , latitude (phi), tilt (beta) and azimuth (gamma). The declination and the
hour angle are treated as input parameters for all cases. Latitude, tilt and azimuth can be given as input parameters
 or left as unknowns (`None`). In total, seven different combinations arise from having these three parameters
as an inputs or as a unknowns. The seven conditionals below correspond to those combinations. The output function `func`
is used as one of the inputs to run_curve_fit which in turn is used to fit the unknowns. The function other outputs is
the 'bounds' tuple containing the bounds for the variables. Bounds for latitude are -90 to 90. Bounds for tilt are 0 to
90. Bounds for azimuth  are -180 to 180. It is noted that, theoretically, bounds for tilt are 0 to 180 (Duffie, John A.,
 and William A. Beckman. Solar engineering of thermal processes. New York: Wiley, 1991.). However a value of tilt >90
 would mean that that the surface has a downward-facing component, which is not the case of the current application.
"""
from pvsystemprofiler.utilities.angle_of_incidence_function import func_costheta
import numpy as np


def select_function(latitude=None, tilt=None, azimuth=None):
    """
    :param latitude: (optional) latitude input value in Degrees.
    :param tilt: (optional) Tilt input value in Degrees.
    :param azimuth: (optional) Azimuth input value in Degrees.
    :return: Customized function 'func' and 'bounds' tuple.
    """

    if latitude is None and tilt is None and azimuth is None:
        func = lambda x, phi, beta, gamma: func_costheta(x, phi, beta, gamma)
    elif latitude is not None and tilt is None and azimuth is None:
        func = lambda x, beta, gamma: func_costheta(
            x, np.deg2rad(latitude), beta, gamma
        )

    elif latitude is None and tilt is not None and azimuth is None:
        func = lambda x, phi, gamma: func_costheta(x, phi, np.deg2rad(tilt), gamma)

    elif latitude is None and tilt is None and azimuth is not None:
        func = lambda x, phi, beta: func_costheta(x, phi, beta, np.deg2rad(azimuth))

    elif latitude is None and tilt is not None and azimuth is not None:
        func = lambda x, phi: func_costheta(
            x, phi, np.deg2rad(tilt), np.deg2rad(azimuth)
        )

    elif latitude is not None and tilt is None and azimuth is not None:
        func = lambda x, beta: func_costheta(
            x, np.deg2rad(latitude), beta, np.deg2rad(azimuth)
        )

    elif latitude is not None and tilt is not None and azimuth is None:
        func = lambda x, gamma: func_costheta(
            x, np.deg2rad(latitude), np.deg2rad(tilt), gamma
        )

    bounds_dict = {
        "latitude": [-np.pi / 2, np.pi / 2],
        "tilt": [0, np.pi / 2],
        "azimuth": [-np.inf, np.inf],
    }
    bounds = []

    if latitude is None:
        bounds.append(bounds_dict["latitude"])
    if tilt is None:
        bounds.append(bounds_dict["tilt"])
    if azimuth is None:
        bounds.append(bounds_dict["azimuth"])

    bounds = tuple(np.transpose(bounds).tolist())

    return func, bounds
