""" Longitude run script
This run script allows to run the longitude_study for multiple sites. The site ids to be evaluated can be provided in
 a csv file. Alternatively, the path to a folder containing the input signals of the sites in separate csv files can be
 provided.  The script provides the option to provided the full path to csv file containing latitude and gmt offset for
 each system for comparison.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from time import time

# TODO: remove pth.append after package is deployed
filepath = Path(__file__).resolve().parents[2]
sys.path.append(str(filepath))
from solardatatools.utilities import progress
from pvsystemprofiler.scripts.modules.script_functions import run_failsafe_pipeline
from pvsystemprofiler.scripts.modules.script_functions import resume_run
from pvsystemprofiler.scripts.modules.script_functions import load_generic_data
from pvsystemprofiler.scripts.modules.script_functions import log_file_versions
from pvsystemprofiler.scripts.modules.script_functions import load_system_metadata
from pvsystemprofiler.scripts.modules.script_functions import generate_list
from solardatatools.dataio import load_cassandra_data
from pvsystemprofiler.scripts.modules.script_functions import extract_sys_parameters
from pvsystemprofiler.scripts.modules.script_functions import get_commandline_inputs
from pvsystemprofiler.scripts.modules.script_functions import run_failsafe_lon_estimation
from pvsystemprofiler.scripts.modules.script_functions import run_failsafe_lat_estimation
from pvsystemprofiler.scripts.modules.script_functions import run_failsafe_ta_estimation
from solardatatools import DataHandler


def evaluate_systems(site_id, inputs_dict, df, site_metadata, json_file_dict=None):
    partial_df_cols = ['site', 'system', 'passes pipeline', 'length', 'capacity_estimate', 'data_sampling',
                       'data quality_score', 'data clearness_score', 'inverter_clipping', 'time_shifts_corrected',
                       'time_zone_correction', 'capacity_changes', 'normal_quality_scores']

    if json_file_dict is not None:
        partial_df_cols.extend(['zip_code', 'real longitude', 'real latitude', 'real tilt', 'real azimuth'])
    if inputs_dict['time_shift_manual']:
        partial_df_cols.append('time_shift_manual')

    partial_df = pd.DataFrame(columns=partial_df_cols)

    ll = len(inputs_dict['power_column_label'])

    if inputs_dict['convert_to_ts']:
        dh = DataHandler(df, convert_to_ts=inputs_dict['convert_to_ts'])
        cols = [el[-1] for el in dh.keys]
    else:
        cols = df.columns

    i = 0
    for col_label in cols:
        if col_label.find(inputs_dict['power_column_label']) != -1:
            system_id = col_label[ll:]
            if system_id in site_metadata['system'].tolist():
                i += 1
                dh = DataHandler(df, convert_to_ts=inputs_dict['convert_to_ts'])
                sys_tag = inputs_dict['power_column_label'] + system_id
                sys_mask = site_metadata['system'] == system_id

                if inputs_dict['time_shift_manual']:
                    time_shift_manual = int(site_metadata.loc[sys_mask, 'time_shift_manual'].values[0])
                    if time_shift_manual == 1:
                        dh.fix_dst()
                else:
                    time_shift_manual = 0

                dh, passes_pipeline = run_failsafe_pipeline(dh, sys_tag, inputs_dict['fix_time_shifts'],
                                                            inputs_dict['time_zone_correction'])
                if passes_pipeline:
                    results_list = [site_id, system_id, passes_pipeline, dh.num_days, dh.capacity_estimate,
                                    dh.data_sampling, dh.data_quality_score, dh.data_clearness_score,
                                    dh.inverter_clipping, dh.time_shifts, dh.tz_correction, dh.capacity_changes,
                                    dh.normal_quality_scores]

                    if inputs_dict['time_shift_manual']:
                        results_list.append(time_shift_manual)
                    if json_file_dict is not None:
                        if system_id in json_file_dict.keys():
                            source_file = json_file_dict[system_id]
                            json_information = extract_sys_parameters(source_file, system_id,
                                                                      inputs_dict['s3_location'])
                        else:
                            json_information = [np.nan] * 4
                        results_list.extend(json_information)

                else:
                    results_list = [site_id, system_id, passes_pipeline] + [np.nan] * (len(results_list) - 3)

                if inputs_dict['estimation'] == 'longitude':
                    if inputs_dict['longitude']:
                        real_longitude = float(site_metadata.loc[sys_mask, 'longitude'])
                    if inputs_dict['gmt_offset'] is not None:
                        gmt_offset = inputs_dict['gmt_offset']
                    else:
                        gmt_offset = float(site_metadata.loc[sys_mask, 'gmt_offset'])
                    results_df, passes_estimation = run_failsafe_lon_estimation(dh, real_longitude, gmt_offset)

                elif inputs_dict['estimation'] == 'latitude':
                    if inputs_dict['latitude']:
                        real_latitude = float(site_metadata.loc[sys_mask, 'latitude'])
                    results_df, passes_estimation = run_failsafe_lat_estimation(dh, real_latitude)

                elif inputs_dict['estimation'] == 'tilt_azimuth':
                    if inputs_dict['estimated_longitude']:
                        longitude_input = float(site_metadata.loc[sys_mask, 'estimated_longitude'])
                    if inputs_dict['estimated_latitude']:
                        latitude_input = float(site_metadata.loc[sys_mask, 'latitude'])
                    if inputs_dict['latitude']:
                        real_latitude = float(site_metadata.loc[sys_mask, 'latitude'])
                    if inputs_dict['tilt']:
                        real_tilt = float(site_metadata.loc[sys_mask, 'tilt'])
                    if inputs_dict['azimuth']:
                        real_azimuth = float(site_metadata.loc[sys_mask, 'azimuth'])
                    if inputs_dict['gmt_offset']:
                        gmt_offset = inputs_dict['gmt_offset']
                    else:
                        gmt_offset = float(site_metadata.loc[sys_mask, 'gmt_offset'])
                    results_df, passes_estimation = run_failsafe_ta_estimation(dh, 1, None, longitude_input,
                                                                               latitude_input, None, None,
                                                                               real_latitude, real_tilt, real_azimuth,
                                                                               gmt_offset)
                if inputs_dict['estimation'] in ['longitude', 'latitude', 'tilt_azimuth']:
                    results_df[partial_df_cols] = results_list
                    partial_df = partial_df.append(results_df)
                elif inputs_dict['estimation'] == 'report':
                    partial_df.loc[len(partial_df)] = results_list
    return partial_df


def main(full_df, inputs_dict, df_system_metadata):
    site_run_time = 0
    total_time = 0
    file_list, json_file_dict = generate_list(inputs_dict, full_df, df_system_metadata)

    if inputs_dict['n_files'] != 'all':
        file_list = file_list[:int(inputs_dict['n_files'])]
    if full_df is None:
        full_df = pd.DataFrame()

    for file_ix, file_id in enumerate(file_list):
        t0 = time()
        msg = 'Site/Accum. run time: {0:2.2f} s/{1:2.2f} m'.format(site_run_time, total_time / 60.0)
        progress(file_ix, len(file_list), msg, bar_length=20)

        if inputs_dict['file_label'] is not None:
            i = file_id.find(inputs_dict['file_label'])
            site_id = file_id[:i]
            mask = df_system_metadata['site'] == site_id.split(inputs_dict['file_label'])[0]
        else:
            site_id = file_id.split('.')[0]

            mask = df_system_metadata['site'] == site_id
        site_metadata = df_system_metadata[mask]

        # TODO: integrate option for other data inputs
        if inputs_dict['data_source'] == 's3':
            df = load_generic_data(inputs_dict['s3_location'], inputs_dict['file_label'], site_id)
        if inputs_dict['data_source'] == 'cassandra':
            df = load_cassandra_data(site_id)

        if not site_metadata.empty:
            partial_df = evaluate_systems(site_id, inputs_dict, df, site_metadata, json_file_dict)
        else:
            partial_df = None

        if not partial_df.empty or partial_df is not None:
            full_df = full_df.append(partial_df)
            full_df.index = np.arange(len(full_df))
            full_df.to_csv(inputs_dict['output_file'])
            t1 = time()
            site_run_time = t1 - t0
            total_time += site_run_time

    msg = 'Site/Accum. run time: {0:2.2f} s/{1:2.2f} m'.format(site_run_time, total_time / 60.0)
    if len(file_list) != 0:
        progress(len(file_list), len(file_list), msg, bar_length=20)
    print('finished')
    return


if __name__ == '__main__':
    """
    :param estimation: Estimation to be performed. Options are 'report', 'longitude', 'latitude', 'tilt_azimuth'
    :param input_site_file:  csv file containing list of sites to be evaluated. 'None' if no input file is provided.
    :param n_files: number of files to read. If 'all' all files in folder are read.
    :param s3_location: Absolute path to s3 location of files.
    :param file_label:  Repeating portion of data files label. If 'None', no file label is used. 
    :param power_column_label: Repeating portion of the power column label. 
    :param output_file: Absolute path to csv file containing report results.
    :param fix_time_shits: String, 'True' or 'False'. Specifies if time shifts are to be 
    fixed when running the pipeline.
    :param time_zone_correction: String, 'True' or 'False'. Specifies if the time zone correction is performed when 
    running the pipeline.
    :param check_json: String, 'True' or 'False'. Check json file for location information.
    :param convert_to_ts: String, 'True' or 'False'. Specifies if conversion to time series is performed when 
    running the pipeline.
    :param system_summary_file: Full path to csv file containing longitude and manual time shift flag for each system,
    None if no file
    provided. 
    :param gmt_offset: String. Single value of gmt offset to be used for all estimations. If None a list with individual
    gmt offsets needs to be provided.
    :param data_source: String. Input signal data source. Options are 's3' and 'cassandra'.
    """

    input_kwargs = sys.argv
    inputs_dict = get_commandline_inputs(input_kwargs)

    log_file_versions('solar-data-tools', active_conda_env='pvi-user')
    log_file_versions('pv-system-profiler')

    full_df = resume_run(inputs_dict['output_file'])

    ssf = inputs_dict['system_summary_file']
    if ssf is not None:
        df_system_metadata = load_system_metadata(df_in=ssf, file_label=inputs_dict['file_label'])
        cols = df_system_metadata.columns
        for param in ['longitude', 'latitude', 'tilt', 'azimuth',
                      'estimated_longitude', 'estimated_latitude',
                      'time_shift_manual']:
            if param in cols:
                inputs_dict[param] = True
            else:
                inputs_dict[param] = False
    else:
        df_system_metadata = None

main(full_df, inputs_dict, df_system_metadata)
