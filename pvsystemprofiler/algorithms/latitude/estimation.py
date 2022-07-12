from pvsystemprofiler.algorithms.latitude.calculation import calculate_latitude


def estimate_latitude(hours_daylight, delta):
    latitude_estimate = calculate_latitude(hours_daylight, delta)
    return latitude_estimate

