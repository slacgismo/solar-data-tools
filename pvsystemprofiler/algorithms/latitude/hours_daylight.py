import numpy as np
from solardatatools.sunrise_sunset import rise_set_rough
from solardatatools.daytime import find_daytime, detect_sun


def calculate_hours_daylight_raw(data_in, data_sampling, threshold=0.001):
    boolean_daytime = find_daytime(data_in, threshold)
    return (np.sum(boolean_daytime, axis=0)) * data_sampling / 60


def calculate_hours_daylight(data_in, threshold=0.001):
    bool_msk = detect_sun(data_in, threshold=threshold)
    measurements = rise_set_rough(bool_msk)
    return measurements['sunsets'] - measurements['sunrises']
