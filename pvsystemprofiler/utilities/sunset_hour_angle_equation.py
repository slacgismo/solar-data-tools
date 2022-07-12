import numpy as np
import pandas as pd
from pvsystemprofiler.utilities.time_convert import solar_to_clock


def sunset_hour_angle(doy, lat):
    b = np.deg2rad((360 / 365) * (doy - 1))
    delta = (0.006918 - 0.399912 * np.cos(b) + 0.070257 * np.sin(b) -
             0.006758 * np.cos(2 * b) + 0.000907 * np.sin(2 * b) -
             0.002697 * np.cos(3 * b) + 0.00148 * np.sin(3 * b))
    sunset_hour_angle = np.arccos(
        -np.tan(np.deg2rad(lat)) * np.tan(delta)
    )
    return np.rad2deg(sunset_hour_angle)


def num_daylight_hours(doy, lat):
    return (2 / 15) * sunset_hour_angle(doy, lat)


def sunrise_sunset_times(lat, lon, doy, gmt_offset, eot='duffie'):
    ss_ha = sunset_hour_angle(doy, lat)
    ss_st = 12 + ss_ha / 15
    sr_st = 12 - ss_ha / 15
    ss_st *= 60
    sr_st *= 60
    ss_ct = solar_to_clock(ss_st, lon, doy, gmt_offset, eot)
    sr_ct = solar_to_clock(sr_st, lon, doy, gmt_offset, eot)
    ss_ct /= 60
    sr_ct /= 60
    output_table = pd.DataFrame(data={
        'day of year': doy,
        'sunrise times': sr_ct,
        'sunset times': ss_ct
    })
    return output_table
