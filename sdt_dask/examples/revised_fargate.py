import glob, os, sys, logging, argparse

import numpy as np
import pandas as pd
from time import gmtime, strftime
from dask import delayed, compute
from dask.distributed import performance_report
from solardatatools import DataHandler
from sdt_dask.dataplugs.S3Bucket_plug import S3Bucket
from sdt_dask.clients.aws.fargate import Fargate

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

log_file = f'../results/rev_far_{options.workers}w-{options.threads}t-{time_stamp}.log'


def _init_logger(level):
    logger = logging.getLogger(__name__)
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

PA_NUMBER = os.getenv("project-pa-number")
TAGS = {
    "project-pa-number": PA_NUMBER,
    "project": "pvinsight"
}
VPC = "vpc-ab2ff6d3"  # for us-west-2
IMAGE = "nimishy/sdt-windows:latest"

AWS_DEFAULT_REGION = os.getenv('AWS_DEFAULT_REGION')
ENVIRONMENT = {
    'AWS_ACCESS_KEY_ID': os.getenv('AWS_ACCESS_KEY_ID'),
    'AWS_SECRET_ACCESS_KEY': os.getenv('AWS_SECRET_ACCESS_KEY')
}

WORKERS = int(options.workers)
THREADS_PER_WORKER = int(options.threads)

bucket = "pvinsight-dask-baseline"
__logger__.info('Grabbing files from S3 bucket: %s', bucket)
data_plug = S3Bucket(bucket)
KEYS = data_plug._pull_keys()
__logger__.info('Grabbed %s files from %s', len(KEYS), bucket)


def run_pipeline(datahandler, key, fix_shifts, verbose=False):
    try:
        datahandler.run_pipeline(fix_shifts=fix_shifts, verbose=verbose)
        __logger__.debug('%s:: run_pipeline Successful', key)
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
    config_client = Fargate(image=IMAGE,
                            tags=TAGS,
                            vpc=VPC,
                            region_name=AWS_DEFAULT_REGION,
                            environment=ENVIRONMENT,
                            n_workers=WORKERS,
                            threads_per_worker=THREADS_PER_WORKER
                            )
    client, cluster = config_client.init_client()
    __logger__.info('Fargate Dask Client Initialized with %s worker(s) and %s thread(s)', WORKERS, THREADS_PER_WORKER)

    with performance_report(
            filename=f"../results/dask-report-fargate-rev-{WORKERS}w-{THREADS_PER_WORKER}t-{time_stamp}.html"):
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

        __logger__.info('Generating Report')
        filename = f'../results/summary_report_fargate_rev_{WORKERS}w_{THREADS_PER_WORKER}t_{time_stamp}.csv'
        df.sort_values(by=['keys'], inplace=True, ascending=True)
        df.to_csv(filename)
        __logger__.info('Creating summary report %s', filename)

        client.shutdown()
        __logger__.info('Dask Client Shutdown')
        cluster.close()
        __logger__.info('Fargate Cluster Closed')

except Exception as e:
    __logger__.exception('%s', e)


