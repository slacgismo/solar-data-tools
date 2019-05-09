import numpy as np


def energy_com(data):
    div1 = np.dot(np.linspace(0, 24, data.shape[0]), data)
    div2 = np.sum(data, axis=0)
    com = np.empty_like(div1)
    com[:] = np.nan
    msk = div2 != 0
    com[msk] = np.divide(div1[msk], div2[msk])
    return com

def avg_sunrise_sunset(data_in):
    data = np.copy(data_in)
    try:
        with np.errstate(invalid='ignore'):
            night_msk = data < 0.005 * np.max(data[~np.isnan(data)])
    except ValueError:
        night_msk = data < 0.005 * np.max(data)
    data[night_msk] = np.nan
    good_vals = (~np.isnan(data)).astype(int)
    sunrise_idxs = np.argmax(good_vals, axis=0)
    sunset_idxs = data.shape[0] - np.argmax(np.flip(good_vals, 0), axis=0)
    sunset_idxs[sunset_idxs == data.shape[0]] = data.shape[0] - 1
    hour_of_day = np.linspace(0, 24, data.shape[0])
    sunrise_times = hour_of_day[sunrise_idxs]
    sunset_times = hour_of_day[sunset_idxs]
    solar_noon = (sunrise_times + sunset_times) / 2
    return solar_noon