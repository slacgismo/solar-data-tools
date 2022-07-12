import os
import sys
import json
import boto3
import subprocess
import paramiko
from smart_open import smart_open
import numpy as np
import pandas as pd
from solardatatools.utilities import progress
from pvsystemprofiler.longitude_study import LongitudeStudy
from pvsystemprofiler.latitude_study import LatitudeStudy
from pvsystemprofiler.tilt_azimuth_study import TiltAzimuthStudy


def get_address(tag_name, region, client):
    """
    Collects the addresses of the aws instances being used for the estimation
    :param tag_name: aws 'Name' tag of the instances
    :param region: aws region
    :param client: aws client
    :return: list with aws instance addresses
    """
    ec2 = boto3.Session(profile_name='default', region_name=region).client(client)
    target_instances = ec2.describe_instances(Filters=[{'Name': 'tag:Name', 'Values': [tag_name]}])

    ec2_instances = []
    for each_instance in target_instances['Reservations']:
        for found_instance in each_instance['Instances']:
            if found_instance['PublicDnsName'] != '':
                ec2_instances.append(found_instance['PublicDnsName'])
    return ec2_instances


def remote_execute(user, instance_id, key, shell_commands, verbose=True):
    """
    Executes a list of bash commands remotely on Amazon Web Services (AWS) instances from another computer.
    :param user: AWS instance user name, i.e. `ubuntu`.
    :param instance_id: AWS public ip address of instance.
    :param key: AWS key. Usually a *.pem file located in the .aws folder.
    :param shell_commands: list of shell commands to be executed remotely.
    :param verbose: provides the output to each remote command execution.
    """
    k = paramiko.RSAKey.from_private_key_file(key)
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(hostname=instance_id, username=user, pkey=k, allow_agent=False, look_for_keys=False)
    command_dict = {}
    for command in shell_commands:
        if verbose:
            print("running command: {}".format(command))
        stdin, stdout, stderr = c.exec_command(command)
        command_dict[command] = [stdout.read(), stderr.read()]
    c.close()
    return command_dict


def copy_to_s3(input_file_name, input_file_location):
    """
    Copies a local file to a Amazon Web Services (AWS) s3 bucket.
    :param input_file_name: name of the file to be copied to the AWS s3 bucket.
    :param input_file_location: full path to the destination AWS s3 bucket.
    """
    bucket, prefix = get_s3_bucket_and_prefix(input_file_location)
    destination_file_name = prefix + '/generated_site_list.csv'

    content = open(input_file_name, 'rb')
    s3 = boto3.client('s3')
    s3.put_object(Bucket=bucket, Key=destination_file_name, Body=content)


def get_s3_bucket_and_prefix(s3_location):
    """
    Splits absolute s3 location into parameters used to copy and execute commands remotely.
    :param s3_location: full path to s3 bucket location.
    :return: s3 bucket and prefix.
    """
    if s3_location[-1] != '/':
        s3_location += '/'
    i = 5
    j = s3_location.find('/', i)
    bucket = s3_location[i:j]
    prefix = s3_location[j + 1:-1]
    return bucket, prefix


def load_generic_data(location, file_label, file_id, extension='.csv', parse_dates=[0], nrows=None):
    """
    Loads csv file containing input signals for a given site.
    :param location: String. absolute path to csv file containing input signals.
    :param file_label: String. Repeating portion of data files label. If 'None', no file label is used.
    :param file_id: String. Individual identifier of a site csv file
    :param extension: String, optional. Extension of file containing input signal.
    :param parse_dates: Optional. 'read_csv' kwarg.
    :param nrows: number of rows from input signal file to be read.
    :return: Dataframe containing input signals for `file_id`.
    """
    if file_label is None:
        to_read = location + file_id + extension
    else:
        to_read = location + file_id + file_label + extension

    if nrows is None:
        df = pd.read_csv(to_read, index_col=0, parse_dates=parse_dates)
    else:
        df = pd.read_csv(to_read, index_col=0, parse_dates=parse_dates, nrows=nrows)
    return df


def get_checked_sites(df):
    """
    Returns list of sites that have already been analyzed
    :param df: pandas dataframe containing results from reports or parameter study.
    :return:  List with sites that have already been analyzed.
    """
    if df is not None:
        checked_sites = df['site'].unique().tolist()
        checked_sites.sort()
    else:
        checked_sites = []
    return checked_sites


def resume_run(results_file):
    """
    Loads the output dataFrame from an incomplete run provided the results csv file name
    :param results_file: full path to csv file containing partial results.
    :return: df dataFrame containing partial results, None if there are no partial results.
    """
    if os.path.isfile(results_file):
        df = pd.read_csv(results_file, index_col=0)
        df['site'] = df['site'].apply(str)
        df['system'] = df['system'].apply(str)
    else:
        df = None
    return df


def enumerate_files(s3_location, extension='.csv', file_size_list=False):
    """
    Returns a list with the file names with a given extension located in a AWS s3 bucket.
    :param s3_location: String. Full path to s3 bucket from which a list of files is to be generated.
    :param extension: String. Extension of the files to be included in `output_list`.
    :param file_size_list: Boolean, generate a list with the size of each file with `extension`.
    :return: `output_list' with file names and (optional) list with site of files in `output list'.
    """
    s3_bucket, prefix = get_s3_bucket_and_prefix(s3_location)
    s3 = boto3.client('s3')
    output_list = []
    size_list = []
    for obj in s3.list_objects_v2(Bucket=s3_bucket, Prefix=prefix)['Contents']:
        if (obj['Key']).find(extension) != -1:
            file_name = obj['Key']
            file_size = obj['Size']
            i = file_name.rfind('/')
            file_name = file_name[i + 1:]
            output_list.append(file_name)
            size_list.append(file_size)
    if file_size_list:
        return output_list, size_list
    else:
        return output_list


def create_system_dict(df):
    """
    Reads a pandas dataFrame and creates a dictionary where the keys are the site ids and the values are a the list of
    systems for each site.
    :param df: pandas dataFrame with site and system information a `site` and a `system` column.
    :return: dictionary with systems associated to each site.
    """
    site_list = df['site'].unique().tolist()
    ss_dict = {}
    for site in site_list:
        systems_in_site = df[df['site'] == site]['system'].values.tolist()
        ss_dict[site] = systems_in_site
    return site_list, ss_dict


def create_json_dict(json_list, location):
    """
    returns a dictionary containing the system ids given a location containing json files with site information.
    :param json_list: list of json files containing site information.
    :param location: absolute path to folder containing json files with site information.
    :return: dictionary containing system ids as keys and site ids as values.
    """
    system_dict = {}
    for file in json_list:
        for line in smart_open(location + file, 'rb'):
            file_json = json.loads(line)
            if len(file_json['Inverters']) != 0:
                for inv_id in file_json['Inverters']:
                    system = file_json['Inverters'][inv_id]['inverter_id']
                    system_dict[system] = file
    return system_dict


def log_file_versions(utility, active_conda_env=None, output_folder_location='./',
                      conda_location='/home/ubuntu/miniconda3/', repository_location='/home/ubuntu/github/'):
    """
    Writes a text file with the utility version as well as other available package information. This utility may have
    been installed in conda using pip or may be in a local GitHub repository.
    :param utility: name of the utility to be logged. Examples are `solar-data-tools' and 'pv-system-profiler'
    :param active_conda_env: optional. Name of the conda environment being used. Examples are pvi-user and `pvi-dev`
    :param output_folder_location: location of the output log files.
    :param conda_location: optional. Full path to the local conda folder location.
    :param repository_location: optional. Location of the GitHub repository.
    :return:
    """
    if active_conda_env is None:
        conda = conda_location + 'bin/conda' + ' '
        conda_env = subprocess.check_output(conda + "env list", shell=True, encoding='utf-8')
        for line in conda_env.splitlines():
            if '*' in line:
                i = line.find('/')
                active_conda_environment = line[i:].split('/')[-1]
    else:
        active_conda_environment = active_conda_env

    output_string = 'active conda environment:' + ' ' + active_conda_environment + '\n'
    pip = conda_location + 'envs/' + active_conda_environment + '/bin/pip'
    try:
        pip_list = subprocess.check_output(pip + ' ' + 'show' + ' ' + utility, shell=True, encoding='utf-8')

        for line in pip_list.splitlines():
            if 'Location' in line:
                i = line.find(':')
                location = line[i + 1:]
            if 'Version' in line:
                i = line.find(':')
                location = line[i + 1:]
                output_string += 'utility version:' + ' ' + location + '\n'
    except:
        pip_list = ''
        output_string += 'utility version:' + ' ' + 'n/a' + '\n'
    try:
        if pip_list.find('site-packages') != -1:
            repository = ''
        else:
            if repository_location is not None:
                location = repository_location + utility
            repository = subprocess.check_output('/usr/bin/git -C' + ' ' + location + ' ' + 'log -n 1', shell=True,
                                                 encoding='utf-8')
        output_string += 'repository location:' + ' ' + location + '\n'

        for line in repository.splitlines():
            if line.find('commit') != - 1:
                i = line.find(' ')
                commit_id = line[i:]
                output_string += 'commit id:' + ' ' + commit_id + '\n'
            if line.find('Author') != - 1:
                i = line.find(': ')
                author = line[i + 1:]
                output_string += 'author:' + ' ' + author + '\n'
            if line.find('Date') != - 1:
                i = line.find(' ')
                date = line[i:]
                output_string += 'date:' + ' ' + date + '\n'
    except:
        output_string += 'commit id:' + ' ' + 'n/a' + '\n'
        output_string += 'author:' + ' ' + 'n/a' + '\n'
        output_string += 'date:' + ' ' + 'n/a' + '\n'

    output_file = open(output_folder_location + utility + '_' + 'versions.log', 'w')
    output_file.write(output_string)
    output_file.close()
    return


def create_system_list(file_label, signal_label, s3_location):
    """
    returns a list of systems present in a `s3_bucket`.
    :param file_label: String. Repeating part of label of files containing input data. For the site list
    ['1_signal.csv', '2_signal.csv'] with `file_label`='_signal'.
    :param signal_label: String. Label of the input signal, i.e. `ac_power_inv_` and `dc_current_inv`.
    :param s3_location: full path to AWS s3 bucket containing csv files with site input signals.
    :return: List containing ids for systems in s3_location.
    """
    bucket, prefix = get_s3_bucket_and_prefix(s3_location)
    file_list = enumerate_files(bucket, prefix)
    ll = len(signal_label)
    system_list = pd.DataFrame(columns=['site', 'system'])

    for file_ix, file_id in enumerate(file_list):
        progress(file_ix, len(file_list), 'Generating system list', bar_length=20)
        file_name = file_id.split('/')[1]
        i = file_name.find(file_label)
        file_id = file_name[:i]
        df = load_generic_data(s3_location, file_label, file_id, nrows=2)
        cols = df.columns
        for col_label in cols:
            if col_label.find(signal_label) != -1:
                system_id = col_label[ll:]
                system_list.loc[len(system_list)] = file_id, system_id
    progress(len(file_list), len(file_list), 'Generating system list', bar_length=20)
    return system_list


def extract_sys_parameters(file_name, system, location):
    """
    Retrieves a list with zip code, inverter id, longitude, latitude, tilt and azimuth for a given system from
    a json file containing site information.
    site.
    :param file_name: String. Name of the json file containing site information.
    :param system: String. system id for which information is to be retrieved from json file
    :param location: String. Full path to folder containing site json file.
    :return: list with system zip code, inverter id, longitude, latitude, tilt and azimuth
    """
    for line in smart_open(location + file_name, 'rb'):
        file_json = json.loads(line)
        parameters = []
        if len(file_json['Inverters']) == 0:
            parameters = [np.nan] * 5
        else:
            try:
                zc = file_json['Site']['location'][-5:]
                zc = '00000' if not zc.isnumeric() else zc
            except KeyError:
                zc = '00000'
            parameters.append(zc)
            for inv_id in file_json['Inverters']:
                mount_id = 'Mount ' + inv_id.split(' ')[1]
                sys_id = file_json['Inverters'][inv_id]['inverter_id']
                if sys_id == system:
                    for coord_ix, coord_id in enumerate(['longitude', 'latitude', 'tilt', 'azimuth']):
                        try:
                            if coord_id in ['longitude', 'latitude']:
                                val = float(file_json['Site'][coord_id])
                            if coord_id in ['tilt', 'azimuth']:
                                val = file_json['Mount'][mount_id][coord_id]
                        except KeyError:
                            val = np.nan
                        val = np.nan if val == '' else val
                        parameters.append(val)
        #file_id = file_json['System']['system_id']
        #parameters.append(file_id)
        return parameters


def siteid_to_filename(sites, file_label, ext='csv'):
    """
    Given a list of site ids returns a list of strings corresponding to the file name obtained by concatenating it to
    the 'file_label` string.  For `sites` in ['1', '2'] with `file_label`='_signal', this method will return the list
    ['1_signal.csv', '2_signal.csv'].
    :param sites: List of strings containing the file id, i.e. non-repeating part of the input signal csv file names.
    :param file_label: String. Repeating part of the input signal csv file name.
    :param ext: String, optional. Input signal file extension
    :return: full file name corresponding to site ids in `sites` list
    """
    checked_sites = []
    for site_id in sites:
        file_name = site_id + file_label + ext
        checked_sites.append(file_name)
    return checked_sites


def filename_to_siteid(sites):
    """
    Given a list of file names containing input signals returns a list of strings with the site ids only. For the list
    of file ids `sites` in ['1_signal.csv', '2_signal.csv'] with `file_label`='_signal', this method will return a list
    with the site ids ['1', '2'].
    .
    :param sites: List of strings containing the full name of csv files containing input signals.
    :param file_label: String. Repeating part of the input signal csv file name.
    :return: list of site ids.
    """
    site_list = []
    for site_id in sites:
        site_id = site_id.split('.')[0]
        site_list.append(site_id)
    return site_list


def run_failsafe_pipeline(dh, sys_tag, fts, tzc):
    """
    Runs the solarDataTools dataHandler pipeline in failsafe mode.
    :param dh: input data handler
    :param sys_tag: Dataframe column label identifying an input signal, i.e. ac_power_01 ar dc_current_02.
    :param fts: Boolean. Fix time shift parameter in `run_pipeline`.
    :param tzc: Boolean. Time zone correction parameter in `run_pipeline`
    :return: Boolean. True if passes pipeline, otherwise False.
    """
    df = dh.data_frame_raw
    try:
        try:
            dh.run_pipeline(power_col=sys_tag, fix_shifts=fts, correct_tz=tzc, verbose=False)
        except ValueError:
            max_val = np.nanquantile(df[sys_tag], 0.95)
            dh.run_pipeline(power_col=sys_tag, fix_shifts=False, correct_tz=tzc, verbose=False, max_val=max_val * 3)
    except:
        return dh, False
    return dh, True


def get_commandline_inputs(input_kwargs):
    """
        :param estimation: Estimation to be performed. Options are 'report', 'longitude', 'latitude', 'tilt_azimuth'.
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
    inputs_dict = {'estimation': input_kwargs[1],
                   'input_site_file': input_kwargs[2] if input_kwargs[2] != 'None' else None,
                   'n_files': input_kwargs[3],
                   's3_location': input_kwargs[4] if input_kwargs[4] != 'None' else None,
                   'file_label': input_kwargs[5] if input_kwargs[5] != 'None' else None,
                   'power_column_label': input_kwargs[6],
                   'output_file': input_kwargs[7],
                   'fix_time_shifts': True if input_kwargs[8] == 'True' else False,
                   'time_zone_correction': True if input_kwargs[9] == 'True' else False,
                   'check_json': True if input_kwargs[10] == 'True' else False,
                   'convert_to_ts': True if input_kwargs[11] == 'True' else False,
                   'system_summary_file': input_kwargs[12] if input_kwargs[12] != 'None' else None,
                   'gmt_offset': input_kwargs[13] if input_kwargs[13] != 'None' else None,
                   'data_source': input_kwargs[14]}
    return inputs_dict


def load_system_metadata(df_in, file_label):
    df = pd.read_csv(df_in, index_col=0)
    df = df[~df['time_shift_manual'].isnull()]
    df['time_shift_manual'] = df['time_shift_manual'].apply(int)
    df = df[df['time_shift_manual'].isin([0, 1])]
    df['site'] = df['site'].apply(str)
    df['system'] = df['system'].apply(str)
    if file_label is not None:
        df['site_file'] = df['site'].apply(lambda x: str(x) + file_label)
    return df


def generate_list(inputs_dict, full_df, df_system_metadata):
    if inputs_dict['s3_location'] is not None:
        full_site_list = enumerate_files(inputs_dict['s3_location'])
        full_site_list = filename_to_siteid(full_site_list)
    else:
        full_site_list = []

    previously_checked_site_list = get_checked_sites(full_df)
    file_list = list(set(full_site_list) - set(previously_checked_site_list))

    if inputs_dict['check_json']:
        json_files = enumerate_files(inputs_dict['s3_location'], extension='.json')
        print('Generating system list from json files')
        json_file_dict = create_json_dict(json_files, inputs_dict['s3_location'])
        print('List generation completed')
    else:
        json_file_dict = None

    if inputs_dict['input_site_file'] is not None:
        input_site_list_df = pd.read_csv(inputs_dict['input_site_file'], index_col=0)
        site_list = input_site_list_df['site'].apply(str)
        site_list = site_list.tolist()
        if len(file_list) != 0:
            file_list = list(set(site_list) & set(file_list))
        else:
            file_list = list(set(site_list))
        if inputs_dict['time_shift_manual']:
            manually_checked_sites = df_system_metadata['site_file'].apply(str).tolist()
            file_list = list(set(file_list) & set(manually_checked_sites))
    file_list.sort()

    return file_list, json_file_dict


def run_failsafe_lon_estimation(dh_in, real_longitude, gmt_offset):
    try:
        runs_lon_estimation = True
        lon_study = LongitudeStudy(data_handler=dh_in, gmt_offset=gmt_offset, true_value=real_longitude)
        lon_study.run(verbose=False)
        p_df = lon_study.results.sort_index().copy()
    except:
        runs_lon_estimation = False
        p_df = pd.DataFrame(columns=['longitude', 'estimator', 'eot_calculation', 'solar_noon_method',
                                     'day_selection_method', 'data_matrix', 'residual', 'site', 'system'])
        p_df.loc[0, :] = np.nan
    return p_df, runs_lon_estimation


def run_failsafe_lat_estimation(dh_in, real_latitude):
    try:
        runs_lat_estimation = True
        lat_study = LatitudeStudy(data_handler=dh_in, lat_true_value=real_latitude)
        lat_study.run()
        p_df = lat_study.results.sort_index().copy()
    except:
        runs_lat_estimation = False
        p_df = pd.DataFrame(columns=['declination_method', 'daylight_calculation', 'data_matrix', 'threshold',
                                     'day_selection_method', 'latitude', 'residual'])
        p_df.loc[0, :] = np.nan
    return p_df, runs_lat_estimation


def run_failsafe_ta_estimation(dh, nrandom, threshold, lon, lat, tilt, azim, real_lat, real_tilt, real_azim,
                               gmt_offset):
    try:
        runs_ta_estimation = True
        ta_study = TiltAzimuthStudy(data_handler=dh, nrandom_init_values=nrandom, daytime_threshold=threshold,
                                    lon_input=lon, lat_input=lat, tilt_input=tilt, azimuth_input=azim,
                                    lat_true_value=real_lat, tilt_true_value=real_tilt, azimuth_true_value=real_azim,
                                    gmt_offset=gmt_offset)
        ta_study.run()
        p_df = ta_study.results.sort_index().copy()
    except:
        runs_ta_estimation = False
        cols = ['day range', 'declination method', 'latitude initial value', 'tilt initial value',
                'azimuth initial value']
        if lat is not None:
            cols.append('latitude')
        if tilt is not None:
            cols.append('tilt')
        if azim is not None:
            cols.append('azimuth')
        if lat is not None:
            cols.append('latitude_residual')
        if tilt is not None:
            cols.append('tilt_residual')
        if azim is not None:
            cols.append('azimuth_residual')
        p_df = pd.DataFrame(columns=cols)
        p_df.loc[0, :] = np.nan
    return p_df, runs_ta_estimation
