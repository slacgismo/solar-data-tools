import sys
import os
import numpy as np
import pandas as pd
import time
import glob
from pathlib import Path

# TODO: remove pth.append after package is deployed
filepath = Path(__file__).resolve().parents[2]
sys.path.append(str(filepath))
from pvsystemprofiler.scripts.modules.config_partitions import get_config
from pvsystemprofiler.scripts.modules.create_partition import create_partition
from pvsystemprofiler.scripts.modules.script_functions import enumerate_files
from pvsystemprofiler.scripts.modules.script_functions import copy_to_s3
from pvsystemprofiler.scripts.modules.script_functions import remote_execute
from pvsystemprofiler.scripts.modules.script_functions import get_address
from pvsystemprofiler.scripts.modules.script_functions import get_commandline_inputs


def build_input_file(s3_location, input_file_location='s3://pv.insight.misc/report_files/'):
    """
    Builds a csv input file by looking at the contents of the s3 bucket containing csv files with signals.
    :param s3_location: aws s3 bucket location of csv files containing signals
    :param input_file_location: s3 bucket location of report files
    :return: DataFrame with signals in a given folder
    """
    site_list, size_list = enumerate_files(s3_location, file_size_list=True)
    site_df = pd.DataFrame()
    site_df['site'] = site_list
    site_df['site'] = site_df['site'].apply(lambda x: x.split('.')[0])
    site_df['file_size'] = size_list
    site_df.to_csv('./generated_site_list.csv')
    copy_to_s3('./generated_site_list.csv', input_file_location)
    return site_df


def get_remote_output_files(partitions, username, destination_dict):
    """
    Collects partition results once estimation is finished.
    :param partitions: String. List containing aws partition addresses.
    :param username:  String. aws user name.
    :param destination_dict: String. Folder where results are saved.
    """
    os.system('mkdir' + ' ' + destination_dict)
    for part_id in partitions:
        get_local_output_file = "scp -i" + part_id.ssh_key_file + ' ' + username + "@" + part_id.public_ip_address + \
                                ":" + part_id.local_output_file + ' ' + destination_dict
        os.system(get_local_output_file)


def combine_results(partitions, destination_dict):
    """
    Combines partitioned results into single csv file
    :param partitions: list containing aws partition addresses
    :param destination_dict: folder where results are saved
    """
    df = pd.DataFrame()
    for part_id in partitions:
        partial_df = pd.read_csv(destination_dict + part_id.local_output_file_name, index_col=0)
        df = df.append(partial_df, ignore_index=True)
        df.index = np.arange(len(df))
    return df


def check_completion(ssh_username, instance_id, ssh_key_file):
    """
    Checks for estimation estimation in partitions
    :param ssh_username: aws username
    :param instance_id: id of the aws instance
    :param ssh_key_file: full path to key file of aws_username
    :return: boolean, True if all partitions are finished
    """
    commands = ["grep -a 'finished' ./out"]
    commands_dict = remote_execute(user=ssh_username, instance_id=instance_id, key=ssh_key_file,
                                   shell_commands=commands, verbose=False)
    if str(commands_dict["grep -a 'finished' ./out"][0]).find('finished') != -1:
        return True
    else:
        return False


def main(estimation, df, ec2_instances, site_input_file, output_folder_location, ssh_key_file, aws_username,
         aws_instance_name, aws_region, aws_client, script_name, script_location, conda_environment, power_column_id,
         convert_to_ts, s3_location, n_files, file_label, fix_time_shifts, time_zone_correction, check_json,
         supplementary_file, data_source, gmt_offset):
    # number of partitions
    n_part = len(ec2_instances)
    total_size = np.sum(df['file_size'])
    # total_size = len(df)
    # size of partition
    part_size = np.ceil(total_size / n_part) * 0.8
    ii = 0
    jj = 0
    partitions = []

    for i in range(n_part):
        local_size = 0
        while local_size < part_size:
            local_size = np.sum(df.loc[ii:jj, 'file_size'])
            if i == n_part - 1:
                jj = len(df) - 1
                local_size = part_size + 1
            jj += 1
        # create partition
        part = get_config(est=estimation, part_id=i, ix_0=ii, ix_n=jj, n_part=n_part, ifl=site_input_file,
                          ofl=output_folder_location, ip_address=ec2_instances[i], skf=ssh_key_file, au=aws_username,
                          ain=aws_instance_name, ar=aws_region, ac=aws_client, script_name=script_name,
                          scripts_location=script_location, conda_env=conda_environment, pcid=power_column_id,
                          cts=convert_to_ts, s3l=s3_location, n_files=n_files, file_label=file_label,
                          fix_time_shifts=fix_time_shifts, time_zone_correction=time_zone_correction,
                          check_json=check_json, sup_file=supplementary_file, data_source=data_source,
                          gmt_offset=gmt_offset)
        # add partition to list
        partitions.append(part)
        create_partition(part)
        ii = jj + 1
        jj = ii

    completion = [False] * len(partitions)
    # check for completion
    while False in completion:
        print(' ')
        for part_ix, part_id in enumerate(partitions):
            if completion[part_ix] is False:
                ssh_key_file = part_id.ssh_key_file
                instance = part_id.public_ip_address
                ssh_username = part_id.aws_username
                new_value = check_completion(ssh_username, instance, ssh_key_file)
                part_id.process_completed = new_value
                completion[part_ix] = new_value
                if new_value is False:
                    status = 'running'
                else:
                    status = 'finished'
                print('partition' + ' ' + str(part_ix) + ':' + ' ' + status)

        time.sleep(10 * 60)
    # collect local result files
    get_remote_output_files(partitions, main_class.aws_username, main_class.global_output_directory)
    # combine results files
    results_df = combine_results(partitions, main_class.global_output_directory)
    # save consolidated results file
    results_df.to_csv(main_class.global_output_file)
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
    :param script_to_execute: Full path to python script to be executed.
    :param conda environment: conda environment used to run script_to_execute.
    :param aws_instance_name: name of amazon web services instances used for running the study. All instances must have
    the same name
    """

    input_kwargs = sys.argv
    inputs_dict = get_commandline_inputs(input_kwargs)
    # The three input arguments below are required in addition to the input arguments required by the run scripts.
    # They are related to 'aws' partition handling.
    script_to_execute = str(sys.argv[-3])
    conda_environment = str(sys.argv[-2])
    aws_instance_name = str(sys.argv[-1])

    estimation = inputs_dict['estimation']
    input_site_file = inputs_dict['input_site_file']
    n_files = inputs_dict['n_files']
    s3_location = inputs_dict['s3_location']
    file_label = inputs_dict['file_label']
    power_column_label = inputs_dict['power_column_label']
    fix_time_shifts = inputs_dict['fix_time_shifts']
    time_zone_correction = inputs_dict['time_zone_correction']
    check_json = inputs_dict['check_json']
    convert_to_ts = inputs_dict['convert_to_ts']
    system_summary_file = inputs_dict['system_summary_file']
    gmt_offset = inputs_dict['gmt_offset']
    data_source = inputs_dict['data_source']

    # Default input variables
    if not input_site_file:
        build_input_file(s3_location)
        input_site_file = 's3://pv.insight.misc/report_files/generated_site_list.csv'

    aws_username = 'ubuntu'
    aws_region = 'us-west-1'
    aws_client = 'ec2'
    output_folder_location = '~/'
    global_output_directory = '~/results/'
    global_output_file = 'results.csv'
    pos = script_to_execute.rfind('/') + 1
    script_location = script_to_execute[:pos]
    script_name = script_to_execute.split('/')[-1]

    # aws licence file
    try:
        ssh_key_file = glob.glob("/Users/*/.aws/*.pem")[0]
    except:
        ssh_key_file = glob.glob("/home/*/.aws/*.pem")[0]

    # create main class
    main_class = get_config(est=estimation, ifl=input_site_file, ofl=output_folder_location, skf=ssh_key_file, au=aws_username,
                            ain=aws_instance_name, ar=aws_region, ac=aws_client, pcid=power_column_label,
                            gof=global_output_file, god=global_output_directory, cts=convert_to_ts,
                            s3l=s3_location, n_files=n_files, file_label=file_label,
                            time_zone_correction=time_zone_correction, check_json=check_json,
                            sup_file=system_summary_file, data_source=data_source, gmt_offset=gmt_offset)
    # collect aws instance addresses
    ec2_instances = get_address(aws_instance_name, aws_region, aws_client)
    # read input site file
    df = pd.read_csv(input_site_file, index_col=0)

    main(estimation, df, ec2_instances, input_site_file, output_folder_location, ssh_key_file, aws_username,
         aws_instance_name, aws_region, aws_client, script_name, script_location, conda_environment, power_column_label,
         convert_to_ts, s3_location, n_files, file_label, fix_time_shifts, time_zone_correction, check_json,
         system_summary_file, data_source, gmt_offset)
