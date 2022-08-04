import numpy as np


def delta_cooper(day_of_year, daily_meas):
    """"
    Declination delta is estimated from equation (1.6.1a) in:
    Duffie, John A., and William A. Beckman. Solar engineering of thermal
    processes. New York: Wiley, 1991.
    """
    delta_1 = 23.45 * np.sin(np.deg2rad(360 * (284 + day_of_year) / 365))
    delta = np.tile(delta_1, (daily_meas, 1))
    return delta


def delta_spencer(day_of_year, daily_meas):
    """"
    Declination delta is estimated from equation (1.6.1b) in:
    Duffie, John A., and William A. Beckman. Solar engineering of thermal
    processes. New York: Wiley, 1991.
    """
    b = np.deg2rad((day_of_year - 1) * 360 / 365)
    delta_1 = (180 / np.pi) * (0.006918 - 0.399912 * np.cos(b) + 0.070257 * np.sin(b) - 0.006758 *
                               np.cos(2 * b) + 0.000907 * np.sin(2 * b) - 0.002697 * np.cos(3 * b) +
                               0.00148 * np.sin(3 * b))
    delta = np.tile(delta_1, (daily_meas, 1))
    return delta
