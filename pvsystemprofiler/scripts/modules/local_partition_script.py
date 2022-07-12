"""
Creates and runs local partitions scripts for run_partition_scripts
"""
import sys
import os
import pandas as pd

# Input parameters
start_index = int(sys.argv[1])
end_index = int(sys.argv[2])
script_name = str(sys.argv[3])
estimation = str(sys.argv[4])
global_input_file = str(sys.argv[5])
local_working_folder = str(sys.argv[6])
local_input_file = str(sys.argv[7])
local_output_file = str(sys.argv[8])
power_column_id = str(sys.argv[9])
convert_to_ts = str(sys.argv[10])
s3_location = str(sys.argv[11])
n_files = str(sys.argv[12])
file_label = str(sys.argv[13])
fix_time_shifts = str(sys.argv[14])
time_zone_correction = str(sys.argv[15])
check_json = str(sys.argv[16])
gmt_offset = str(sys.argv[17])
data_type = str(sys.argv[18])
supplementary_file = str(sys.argv[19])
python_command = str(sys.argv[20])

# read full list of systems save a local copy of systems corresponding to partition
df_full = pd.read_csv(global_input_file, index_col=0)
df_part = df_full.copy()
df_part = df_part[start_index:end_index]
df_part.to_csv(local_input_file)

# create execute command to run study or site report
command = 'setsid nohup' + ' ' + python_command + ' ' + script_name
# create arguments to run `command` with
arguments = estimation + ' ' \
            + local_input_file + ' '  \
            + n_files + ' ' \
            + s3_location + ' ' \
            + file_label + ' ' \
            + power_column_id + ' ' \
            + local_output_file + ' ' \
            + fix_time_shifts + ' '\
            + time_zone_correction + ' '\
            + check_json + ' ' \
            + convert_to_ts + ' ' \
            + supplementary_file + ' ' \
            + gmt_offset + ' ' \
            + data_type

full_command = command + ' ' + arguments + '>out &'
# save local copy of run script
file1 = open(local_working_folder + 'run_local_partition.sh', "w")
print('Running local script')
file1.write('#!/bin/sh\n')
file1.write(full_command)
file1.close()

# execute local run script
os.system(full_command)
