from pvsystemprofiler.algorithms.longitude.calculation import calculate_longitude
from pvsystemprofiler.algorithms.longitude.fitting import fit_longitude


def estimate_longitude(estimator, eot, solarnoon, days, gmt_offset):
    if estimator == 'calculated':
        return calculate_longitude(eot, solarnoon, days, gmt_offset)
    else:
        loss = estimator.split('_')[-1]
        return fit_longitude(eot, solarnoon, days, gmt_offset, loss=loss)
