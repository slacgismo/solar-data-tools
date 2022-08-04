import numpy as np


def determine_keys(latitude, tilt, azimuth):
    key = []
    if latitude is None:
        key.append('latitude_estimate')
    if tilt is None:
        key.append('tilt_estimate')
    if azimuth is None:
        key.append('azimuth_estimate')
    return key


def select_init_values(bounds_dict, dict_keys):
    init_vals = []
    ivr = []
    if 'latitude_estimate' in dict_keys:
        init_vals.append(bounds_dict['latitude'])
        ivr.append(bounds_dict['latitude'])
    else:
        ivr.append(np.nan)
    if 'tilt_estimate' in dict_keys:
        init_vals.append(bounds_dict['tilt'])
        ivr.append(bounds_dict['tilt'])
    else:
        ivr.append(np.nan)
    if 'azimuth_estimate' in dict_keys:
        init_vals.append(bounds_dict['azimuth'])
        ivr.append(bounds_dict['azimuth'])
    else:
        ivr.append(np.nan)
    return init_vals, ivr
