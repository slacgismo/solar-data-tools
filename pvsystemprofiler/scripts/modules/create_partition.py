import sys
from pathlib import Path

# TODO: remove pth.append after package is deployed
filepath = Path(__file__).resolve().parents[1]
sys.path.append(str(filepath))
from pvsystemprofiler.scripts.modules.script_functions import remote_execute


def create_partition(partition):
    """
    Creates remote partition from `run_partition_script`
    :param partition: object containing information about each individual partition
    """
    start_index = partition.ix_0
    end_index = partition.ix_n
    global_input_file = partition.input_file_location
    local_input_file = partition.local_input_file
    local_working_folder = partition.local_working_folder
    local_output_file = partition.local_output_file
    script_name = partition.script_name
    estimation = partition.estimation
    scripts_location = partition.scripts_location
    local_script = scripts_location + 'modules/local_partition_script.py'
    conda_env = partition.conda_environment
    instance = partition.public_ip_address
    ssh_username = partition.aws_username
    ssh_key_file = partition.ssh_key_file
    power_column_id = partition.power_column_id
    convert_to_ts = partition.convert_to_ts
    s3_location = partition.s3_location
    n_files = partition.n_files
    file_label = partition.file_label
    fix_time_shifts = partition.fix_time_shifts
    time_zone_correction = partition.time_zone_correction
    check_json = partition.check_json
    supplementary_file = partition.supplementary_file
    data_type = partition.data_source
    gmt_offset = partition.gmt_offset

    # prepare python command to run local partition
    # extract conda installation folder from local .bashrc
    grep_conda = "grep '__conda_setup=' .bashrc"
    commands = [grep_conda]
    output = remote_execute(ssh_username, instance, ssh_key_file, commands)
    conda_location = output[grep_conda][0]
    conda_location = conda_location.decode('utf-8')
    i = conda_location.find("('")
    j = conda_location.find("bin", i + 2)
    conda_location = conda_location[i + 2: j]
    python_command = conda_location + 'envs/' + conda_env + '/bin/python'

    # delete existing remote partitions
    commands = ['rm estimation* -rf']
    output = remote_execute(ssh_username, instance, ssh_key_file, commands)

    # check for previously created remote folders
    commands = ['ls' + ' ' + local_working_folder]
    output = remote_execute(ssh_username, instance, ssh_key_file, commands)

    # create remote partition if does not exist
    if str(output[commands[0]][1]).find('No such file or directory') != -1:
        # prepare commands to create partition
        commands = ['rm estimation* -rf',
                    'mkdir' + ' ' + local_working_folder,
                    python_command + ' ' + local_script + ' '
                    + str(start_index) + ' '
                    + str(end_index) + ' '
                    + script_name + ' '
                    + estimation + ' '
                    + global_input_file + ' '
                    + local_working_folder + ' '
                    + local_input_file + ' '
                    + local_output_file + ' '
                    + power_column_id + ' '
                    + str(convert_to_ts) + ' '
                    + s3_location + ' '
                    + n_files + ' '
                    + str(file_label) + ' '
                    + str(fix_time_shifts) + ' '
                    + str(time_zone_correction) + ' '
                    + str(check_json) + ' '
                    + str(gmt_offset) + ' '
                    + data_type + ' '
                    + supplementary_file + ' '
                    + python_command
                    ]
    else:
        # if remote partition exist from previous run, resume run
        commands = [local_working_folder + 'run_local_partition.sh']
    # execute remote partition script
    remote_execute(ssh_username, instance, ssh_key_file, commands)
