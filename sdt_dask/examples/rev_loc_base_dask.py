"""
Local test script for Dask using local data plug

It takes in the following arguments:
    -l:string   log level         (default='warning')
    -w:int      worker number     (default=4)
    -t:int      thread number     (default=2)
    -v:bool     verbose           (default=False)
    -r:string   result path       (default='../results/')

Example command to run this test script:
    python rev_loc_base_dask.py -l info -w 2 -t 2
"""
import glob, os, sys, logging, argparse

from time import gmtime, strftime
from sdt_dask.dataplugs.csv_plug import LocalFiles
from sdt_dask.clients.local import Local
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
    "-m",
    "--memory",
    default=5,
    help=(
        "Declare memory limit per worker. "
        "Example --memory 5', default=5"),
)

parser.add_argument(
    "-v",
    "--verbose",
    default=False,
    action='store_true',
    help=(
        "Enable verbose for the dask client. "
        "Example --verbose"),
)

parser.add_argument(
    "-r",
    "--result_path",
    default="../results/",
    help=(
        "Provide the result path. "
        "Example --result ../results/', default='../results/'"),
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
# formats the loggers as well, formatter doesn't support color logs
def _init_logger(level):
    logger = logging.getLogger(__name__)
    # logger.setLevel(level=level)
    logging.basicConfig(filename=log_file,
                        format='%(asctime)s:%(levelname)s:%(name)s:'
                               '%(module)s: %(message)s',
                        encoding='utf-8',
                        level=level)
    handler = logging.StreamHandler(sys.stdout)

    formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:'
                                  '%(module)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Intiialize the logger with level '-l/--level' parsed argument
_init_logger(level)
__logger__ = logging.getLogger(__name__)
__logger__.info('Code started in %s', os.getcwd())
__logger__.info('Saving Logs to %s', log_file)

__logger__.debug('arguments: %s', vars(options))

WORKERS = int(options.workers)
THREADS_PER_WORKER = int(options.threads)
MEMORY = float(options.memory)
VERBOSE = bool(options.verbose)

#  TODO: Change the path
path = "C:\\Users\\Zander\\Documents\\spw_sensor_0\\"

__logger__.info('Grabbing files from path: %s', path)

# Defined Local CSV dataplug
data_plug = LocalFiles(path_to_files=path)
KEYS = [(os.path.basename(fname)[:-4],) for fname in glob.glob(path + "*")]
__logger__.info('Grabbed %s files from %s', len(KEYS), path)

# Sets the dask local client and dask tool
# Uses the dask tool for computation
# The conditional statement is required to allow
# process spawning
if __name__ == '__main__':
    try:
        # Dask Local client Setup
        client_setup = Local(
            n_workers = WORKERS,
            threads_per_worker = THREADS_PER_WORKER,
            memory_per_worker = MEMORY,
            verbose = VERBOSE
        )
        # Dask Local Client Initialization
        client = client_setup.init_client()
        __logger__.info('Local Dask Client Initialized with %s worker(s),'
                        ' %s thread(s) and %s GiB memory per worker',
                        WORKERS, THREADS_PER_WORKER, MEMORY)

        # Dask Tool initialization and set up
        dask_tool = Runner(data_plug=data_plug,
                            client=client,
                            output_path=f"{options.result_path}/{options.workers}w-{options.threads}t/")
        dask_tool.set_up(KEYS, fix_shifts=True, verbose=VERBOSE)

        # Dask Tool Task Compute
        dask_tool.get_result()
    except Exception as e:
        __logger__.exception(e)