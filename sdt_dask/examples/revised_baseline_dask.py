import glob, os, sys, logging, argparse

import numpy as np
import pandas as pd
from time import gmtime, strftime
from dask import delayed, compute
from dask.distributed import performance_report
from solardatatools import DataHandler
from sdt_dask.dataplugs.csv_plug import LocalFiles
from sdt_dask.clients.local import Local
from sdt_dask.dask_tool.sdt_dask import SDTDask

os.system('cls')
time_stamp = strftime("%Y%m%d-%H%M%S", gmtime())

parser = argparse.ArgumentParser()
parser.add_argument(
    "-log",
    "--log",
    default="warning",
    help=(
        "Provide logging level. "
        "Example --log debug', default='warning'"),
)

parser.add_argument(
    "-workers",
    "--workers",
    default=4,
    help=(
        "Declare number of workers. "
        "Example --workers 3', default=4"),
)

parser.add_argument(
    "-threads",
    "--threads",
    default=2,
    help=(
        "Declare number of threads per worker. "
        "Example --threads 3', default=2"),
)

parser.add_argument(
    "-memory",
    "--memory",
    default=5,
    help=(
        "Declare memory limit per worker. "
        "Example --memory 5', default=5"),
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
log_file = f'../results/rev_win_{options.workers}w-{options.threads}t-{time_stamp}.log'

WORKERS = int(options.workers)
THREADS_PER_WORKER = int(options.threads)
MEMORY = float(options.memory)

def _init_logger(level):
    logger = logging.getLogger(__name__)
    # logger.setLevel(level=level)
    logging.basicConfig(filename=log_file,
                        format='%(asctime)s:%(levelname)s:%(name)s:%(module)s: %(message)s',
                        encoding='utf-8',
                        level=level)
    handler = logging.StreamHandler(sys.stdout)

    formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(module)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

_init_logger(level)
__logger__ = logging.getLogger(__name__)
__logger__.info('Code started in %s', os.getcwd())
__logger__.info('Saving Logs to %s', log_file)

#  TODO: Change the path
path = "C:\\Users\\Zander\\Documents\\spw_sensor_0\\"
__logger__.info('Grabbing files from path: %s', path)
data_plug = LocalFiles(path_to_files=path)
KEYS = [(os.path.basename(fname)[:-4],) for fname in glob.glob(path + "*")]
__logger__.info('Grabbed %s files from %s', len(KEYS), path)

if __name__ == '__main__':
    print(KEYS)
    client_setup = Local(
        n_workers = WORKERS,
        threads_per_worker = THREADS_PER_WORKER,
        memory_per_worker = MEMORY,
        verbose = True
    )
    client = client_setup.init_client()
    __logger__.info('Local Dask Client Initialized with %s worker(s), %s thread(s) and %s GiB memory per worker',
                    WORKERS, THREADS_PER_WORKER, MEMORY)
    dask_tool = SDTDask(data_plug=data_plug, client=client, output_path="../results/")
    dask_tool.set_up(KEYS, fix_shifts=True, verbose=True)

    dask_tool.get_result()

