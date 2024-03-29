import glob, os, sys, logging, argparse

import numpy as np
import pandas as pd
from time import gmtime, strftime
from dask import delayed, compute
from dask.distributed import performance_report
from solardatatools import DataHandler
from sdt_dask.dataplugs.csv_plug import LocalFiles
from sdt_dask.clients.local import Local

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


path = "C:\\Users\\Zander\\Documents\\spw_sensor_0\\"
__logger__.info('Grabbing files from path: %s', path)
data_plug = LocalFiles(path_to_files=path)
KEYS = [os.path.basename(fname)[:-4] for fname in glob.glob(path + '*')]
__logger__.info('Grabbed %s files from %s', len(KEYS), path)

def run_pipeline(datahandler, key, fix_shifts, verbose=False):
    try:
        datahandler.run_pipeline(fix_shifts=fix_shifts, verbose=verbose)
        __logger__.info('%s:: run_pipeline Successful', key)
        return datahandler
    except Exception as e:
        __logger__.exception('%s:: run_pipeline: Exception: %s', key, e)
        return datahandler

@delayed
def get_report(datahandler, key):
    if datahandler._ran_pipeline:
        report = datahandler.report
        report = report(return_values=True, verbose=False)
        __logger__.debug("%s:: Report: Successful", key)
        return report
    else:
        __logger__.warning('%s:: Report: failed to run_pipeline', key)
        return {}

@delayed
def get_runtime(datahandler, key):
    if datahandler._ran_pipeline:
        runtime = datahandler.total_time
        __logger__.debug("%s:: Runtime: Successful", key)
        return runtime
    else:
        __logger__.warning('%s:: Runtime: failed to run_pipeline', key)
        return None

reports = []
runtimes = []

if __name__ == '__main__':
    for key in KEYS:
        df = delayed(data_plug.get_data)((key,))
        dh = delayed(DataHandler)(df)
        dh_run = delayed(run_pipeline)(dh, fix_shifts=True, verbose=False, key=key)
        reports.append(get_report(dh_run, key))
        runtimes.append(get_runtime(dh_run, key))

    df_reports = delayed(pd.DataFrame)(reports)
    df_reports = delayed(df_reports.assign)(runtime=runtimes, keys=KEYS)

    # Visualizing the graph
    df_reports.visualize()
    try:
        config_client = Local(
            n_workers = WORKERS,
            threads_per_worker = THREADS_PER_WORKER,
            memory_per_worker = MEMORY,
            verbose = True
        )
        client = config_client.init_client()
        __logger__.info('Local Dask Client Initialized with %s worker(s), %s thread(s) and %s GiB memory per worker',
                        WORKERS, THREADS_PER_WORKER, MEMORY)

        with performance_report(filename=f"../results/dask-report-windows-rev-{WORKERS}w-{THREADS_PER_WORKER}t-{MEMORY}g-{time_stamp}.html"):
            __logger__.info('Starting Computation')
            summary_table = client.compute(df_reports)
            df = summary_table.result()

            scheduler_logs = client.get_scheduler_logs()
            __logger__.info('Scheduler Logs:')
            for log in scheduler_logs:
                __logger__.info('%s', log)
            worker_logs = client.get_worker_logs()
            __logger__.info('Worker Logs:')
            for items, keys in worker_logs.items():
                __logger__.info('%s', items)
                for key in keys:
                    __logger__.info('%s', key)

            filename = f'../results/summary_report_windows_rev_{WORKERS}w_{THREADS_PER_WORKER}t_{MEMORY}g_{time_stamp}.csv'
            df.sort_values(by=['keys'], inplace=True, ascending=True)
            df.to_csv(filename)
            __logger__.info('Creating summary report %s', filename)


        client.shutdown()
        __logger__.info('Dask Client Shutdown')
    except Exception as e:
        __logger__.exception('%s', e)
