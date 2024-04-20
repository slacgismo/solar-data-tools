"""
Code for fargate baseline test

Uses the local csv dataplug
Uses the SDTDask Tool
Uses the Fargate client plug

IMPORTANT:
DO NOT UPLOAD/SHARE THE LOG FILES AS THEY CONTAIN SENSITIVE
INFORMATION INCLUDING THE AWS_ACCESS_KEY_ID.

THE NORMAL LOGS CONTAIN THE ACCESS KEY ID
THE DEBUG LOGS CONTAIN ALL INFORMATION INCLUDING CREDENTIALS

Notice:
The dask tool report and summary will get overwritten,
please rename or save those files separately before
rerunning this code.

The -l/--log option sets the log level (string)
    Options: debugs, info, warning, error, critical, warn

The -w/--workers option sets the number of workers for the
local dask client

The -t/--threads option sets the number of threads per
worker for the dask client

The -v/--verbose option sets the verbose flag for the run_pipeline
function. This information allows the user to  debug the run_pipeline.

TODO:
    Check Environment variables in from line no. XXX - XXX
Example:
    python .\rev_far_base_dask.py -l info -w 1 -t 1 -v
OR
    python rev_far_base_dask.py -l info -w 1 -t 1
"""
import glob, os, sys, logging, argparse

from time import strftime
from sdt_dask.dataplugs.S3Bucket_plug import S3Bucket
from sdt_dask.clients.aws.fargate import Fargate
from sdt_dask.dask_tool.runner import Runner
from sdt_dask.dataplugs.pvdb_plug import PVDBPlug
from sdt_dask.dataplugs.csv_plug import LocalFiles

import pandas as pd

time_stamp = strftime("%Y%m%d-%H%M%S")

"""
Parser Implementation for the following:
  log level   (default='warning')
  workers     (default=4)
  threads     (default=2)
  verbose     (default=False)
"""
parser = argparse.ArgumentParser()
parser.add_argument(
    "-l",
    "--log",
    default="warning",
    help=(
        "Provide logging level. "
        "Example --log debug', default='warning'"),
)

parser.add_argument(
    "-w",
    "--workers",
    default=4,
    help=(
        "Declare number of workers. "
        "Example --workers 3', default=4"),
)

parser.add_argument(
    "-t",
    "--threads",
    default=2,
    help=(
        "Declare number of threads per worker. "
        "Example --threads 3', default=2"),
)

parser.add_argument(
    "-v",
    "--verbose",
    default=False,
    action='store_true',
    help=(
        "Enable verbose for run_pipeline. "
        "Example --verbose"),
)
options = parser.parse_args()
levels = {
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warning': logging.WARNING,
    'error': logging.ERROR,
    'critical': logging.CRITICAL,
    'warn': logging.WARN
}
level = levels[options.log.lower()]
log_file = (f"../results/rev_far_{options.workers}w-"
            f"{options.threads}t-{time_stamp}.log")


# Function for the logger handler and formatter for this file
# formats the loggers as well, No color logs available
def _init_logger(level):
    logger = logging.getLogger(__name__)
    # logger.setLevel(level=level)
    logging.basicConfig(filename=log_file,
                        format='%(asctime)s:%(levelname)s:%(name)s:'
                               '%(module)s: %(message)s',
                        encoding='utf-8',
                        level=level)
    handler = logging.StreamHandler(sys.stdout)

    formatter = logging.Formatter(
        '%(asctime)s:%(levelname)s:%(name)s:%(module)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


# Intiialize the logger with level '-l/--level' parsed argument
_init_logger(level)
__logger__ = logging.getLogger(__name__)
__logger__.info('Code started in %s', os.getcwd())
__logger__.info('Saving Logs to %s', log_file)

__logger__.debug('arguments: %s', vars(options))

# TODO:
#   Verify and change the environment variables
PA_NUMBER = os.getenv("project_pa_number")
TAGS = {
    "project-pa-number": PA_NUMBER,
    "project": "pvinsight"
}
VPC = "vpc-ab2ff6d3"  # for us-west-2
IMAGE = "nimishy/sdt-windows:latest"
IMAGE = "nimishy/sdt-cloud-win:latest"
IMAGE = "nimishy/p_3.10.11_dask:latest"

AWS_DEFAULT_REGION = os.getenv('AWS_DEFAULT_REGION')
ENVIRONMENT = {
    'AWS_ACCESS_KEY_ID': os.getenv('AWS_ACCESS_KEY_ID'),
    'AWS_SECRET_ACCESS_KEY': os.getenv('AWS_SECRET_ACCESS_KEY')
}

__logger__.debug('Environment: %s', ENVIRONMENT)
__logger__.debug('Tags: %s', TAGS)

WORKERS = int(options.workers)
THREADS_PER_WORKER = int(options.threads)
VERBOSE = bool(options.verbose)

# Defined PVDB dataplug
data_plug = PVDBPlug()

# read keys from files
site_num_file = "./pvdb_data/num_sensors_per_site_ac_power.csv"
site_sensor_file = "./pvdb_data/all_sensors_per_site_ac_power.csv"
df = pd.read_csv(site_num_file)
KEYS = []
for row in df.itertuples():
    site_id = row.site
    num = row.number
    for i in range(num):
        KEYS.append((site_id, i))

# Sets the dask fargate client and dask tool
# Uses the dask tool for computation
if __name__ == '__main__':
    try:
        # Dask Fargate client Setup
        client_setup = Fargate(image=IMAGE,
                               tags=TAGS,
                               vpc=VPC,
                               region_name=AWS_DEFAULT_REGION,
                               environment=ENVIRONMENT,
                               n_workers=WORKERS,
                               threads_per_worker=THREADS_PER_WORKER
                               )
        # Dask Local Client Initialization
        client = client_setup.init_client()
        __logger__.info('Fargate Dask Client Initialized with %s worker(s)'
                        ' and %s thread(s)',
                        WORKERS, THREADS_PER_WORKER)

        # Dask Tool initialization and set up
        dask_tool = Runner(data_plug=data_plug,
                            client=client,
                            output_path="../results/")
        dask_tool.set_up(KEYS, fix_shifts=True, verbose=VERBOSE)

        # Dask Tool Task Compute
        output_html = f"pvdb_rev_far_dask-report_{options.workers}w-{options.threads}t-{time_stamp}.html"
        output_csv = f"pvdb_rev_far_summary_report_{options.workers}w-{options.threads}t-{time_stamp}.csv"
        
        sensor_df = pd.read_csv(site_sensor_file)
        dask_tool.get_result(dask_report = output_html, summary_report = output_csv, additional_columns = sensor_df.iloc[:, :2])
        
    except Exception as e:
        __logger__.exception(e)
