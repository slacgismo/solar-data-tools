""" Time Conversion Module
Convert between solar time and clock time, given a known longitude. This is the
definition of solar time and standard time, given in equation (1.5.2) in [1].

    [1] Duffie, John A., and William A. Beckman. Solar engineering of thermal
        processes. New York: Wiley, 1991.
"""

from pvsystemprofiler.utilities.equation_of_time import eot_da_rosa, eot_duffie


def solar_to_clock(solar_time, lon, doy, gmt_offset, eot='duffie'):
    """
    :param solar_time: solar time in minutes since midnight (float or array)
    :param lon: longitude (float)
    :param doy: day of year (float or array)
    :param gmt_offset: local timezone offset in hours from UTC/GMT (float or int)
    :param eot: string specifying which equation of time formulation to use
    :return:
    """
    if eot.lower() in ('duffie', 'd'):
        eot = eot_duffie(doy)
    elif eot.lower() in ('da_rosa', 'dr'):
        eot = eot_da_rosa(doy)
    else:
        print('Please select either Duffie or Da Rosa for the equation of time')
        return
    st = solar_time
    ct = st - eot - 4 * (lon - 15 * gmt_offset)
    return ct


def clock_to_solar(clock_time, lon, doy, gmt_offset, eot='duffie'):
    if eot.lower() in ('duffie', 'd'):
        eot = eot_duffie(doy)
    elif eot.lower() in ('da_rosa', 'dr'):
        eot = eot_da_rosa(doy)
    else:
        print('Please select either Duffie or Da Rosa for the equation of time')
        return
    ct = clock_time
    st = ct + eot + 4 * (lon - 15 * gmt_offset)
    return st
