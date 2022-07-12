from solardatatools.algorithms import SunriseSunset


def get_optimized_sunrise_sunset(filled_data_matrix=None, raw_data_matrix=None):
    optimized_dict = {}
    ss = SunriseSunset()
    if raw_data_matrix is not None:
        ss.run_optimizer(data=raw_data_matrix)
        optimized_dict['est_sr_raw'] = ss.sunrise_estimates
        optimized_dict['est_ss_raw'] = ss.sunset_estimates
        optimized_dict['meas_sr_raw'] = ss.sunrise_measurements
        optimized_dict['meas_ss_raw'] = ss.sunset_measurements
        optimized_dict['thres_raw'] = ss.threshold
    else:
        optimized_dict['est_sr_raw'] = None
        optimized_dict['est_ss_raw'] = None
        optimized_dict['meas_sr_raw'] = None
        optimized_dict['meas_ss_raw'] = None
        optimized_dict['thres_raw'] = None

    if filled_data_matrix is not None:
        ss.run_optimizer(data=filled_data_matrix)
        optimized_dict['est_sr_f'] = ss.sunrise_estimates
        optimized_dict['est_ss_f'] = ss.sunset_estimates
        optimized_dict['meas_sr_f'] = ss.sunrise_measurements
        optimized_dict['meas_ss_f'] = ss.sunset_measurements
        optimized_dict['thres_f'] = ss.threshold
    else:
        optimized_dict['est_sr_f'] = None
        optimized_dict['est_ss_f'] = None
        optimized_dict['meas_sr_f'] = None
        optimized_dict['meas_ss_f'] = None
        optimized_dict['thres_f'] = None
    return optimized_dict
