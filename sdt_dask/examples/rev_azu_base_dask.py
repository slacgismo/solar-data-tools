"""
Azure test script for Dask using S3Bucket data plug

It takes in the following arguments:
    -l:string   log level         (default='warning')
    -w:int      worker number     (default=4)
    -t:int      thread number     (default=2)
    -v:bool     verbose           (default=False)
    -b:string   bucket name       (default='pvinsight-dask-baseline')
    -r:string   result path       (default='../results/')

Example command to run this test script:
    python rev_azu_base_dask.py -l info -w 2 -t 2

Before running the script, make sure to set up the environment variables for Azure:
    RESOURCE_GROUP
    VNET
    SECURITY_GROUP
"""

import os, sys, logging, argparse

from time import strftime
from sdt_dask.clients.azure.azure import Azure
from sdt_dask.dataplugs.S3Bucket_plug import S3Bucket
from sdt_dask.dask_tool.runner import Runner

time_stamp = strftime("%Y%m%d-%H%M%S")

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

parser.add_argument(
    "-b",
    "--bucket",
    default="pvinsight-dask-baseline",
    help=(
        "Provide the bucket name. "
        "Example --bucket pvinsight-dask-baseline', default='pvinsight-dask-baseline'"),
)

parser.add_argument(
    "-r",
    "--result_path",
    default="../results/",
    help=(
        "Provide the result path. "
        "Example --result_path ../results/', default='../results/'"),
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

# check whether result path exists, if not create it
if not os.path.exists(f"{options.result_path}/{options.workers}w-{options.threads}t/"):
    os.makedirs(f"{options.result_path}/{options.workers}w-{options.threads}t/")
log_file = (f"{options.result_path}/{options.workers}w-{options.threads}t/rev_azu_{options.workers}w-"
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

resource_group = os.getenv("RESOURCE_GROUP")
vnet = os.getenv("VNET")
security_group = os.getenv("SECURITY_GROUP")
image = "nimishy/sdt-cloud-win:latest"

__logger__.info('Resource Group: %s', resource_group)
__logger__.info('VNet: %s', vnet)
__logger__.info('Security Group: %s', security_group)
__logger__.info('Docker Image: %s', image)

WORKERS = int(options.workers)
THREADS_PER_WORKER = int(options.threads)
VERBOSE = bool(options.verbose)

bucket = options.bucket

__logger__.info('Grabbing files from bucket: %s', bucket)

# Defined S3 Bucket dataplug
data_plug = S3Bucket(bucket_name=bucket)

# Required for S3 Bucket pull keys as a list given as output
key_list = data_plug._pull_keys()
KEYS = [(key,) for key in key_list]
__logger__.info('Grabbed %s files from %s', len(KEYS), bucket)

# Sets the dask fargate client and dask tool
# Uses the dask tool for computation
if __name__ == '__main__':
    try:
        # Dask Fargate client Setup
        worker_options = {
            "nthreads": THREADS_PER_WORKER,
            "memory_limit": "15.63GiB" # using the maximum memory limit for Azure VM
        }
        client_setup = Azure(
            resource_group=resource_group,
            vnet=vnet,
            security_group=security_group,
            n_workers=WORKERS,
            worker_options=worker_options,
            docker_image=image
        )
        # Dask Local Client Initialization
        client = client_setup.init_client()

        __logger__.info('Azure Dask Client Initialized with %s worker(s)'
                        ' and %s thread(s)',
                        WORKERS, THREADS_PER_WORKER)

        # Dask Tool initialization and set up
        dask_tool = Runner(data_plug=data_plug,
                            client=client,
                            output_path=f"{options.result_path}/{options.workers}w-{options.threads}t/")
        dask_tool.set_up(KEYS, fix_shifts=True, verbose=VERBOSE)

        # Dask Tool Task Compute
        dask_tool.get_result()
    except Exception as e:
        __logger__.exception(e)
