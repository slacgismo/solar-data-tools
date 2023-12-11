import os
from solardatatools import DataHandler
import pandas as pd
import boto3
from dask import delayed, compute
from dask.distributed import Client, LocalCluster, performance_report
import json
import click
import tempfile
from solardatatools.dataio import load_cassandra_data


def s3_csv_to_dh(path):
    """
    Converts a s3 CSV file into a solar-data-tools DataHandler.
    Parameters:
    - file: Path to the CSV file.
    Returns:
    - A tuple of the file name and its corresponding DataHandler.
    """
    df = pd.read_csv(path, index_col=0)
    # Convert index from int to datetime object
    df.index = pd.to_datetime(df.index)
    dh = DataHandler(df)
    name = path.split("/")[-1].removesuffix(".csv")
    return (name, dh)


def get_csvs_in_s3(s3, bucket, prefix):
    """
    Gets the csvs in an s3 directory.
    Parameters:
    - s3: Boto3 s3 client
    - bucket: Bucket containing the csvs.
    - prefix: Prefix appended to the bucket name when searching for files
    Returns:
    - An array of the csv file paths.
    """
    paths = []
    data_bucket = s3.Bucket(bucket)
    for object_summary in data_bucket.objects.filter(Prefix=prefix):
        if object_summary.key.endswith(".csv"):
            file = f"s3://{bucket}/{object_summary.key}"
            paths.append(file)
    return paths


def local_csv_to_dh(file):
    """
    Converts a local CSV file into a solar-data-tools DataHandler.
    Parameters:
    - file: Path to the CSV file.
    Returns:
    - A tuple of the file name and its corresponding DataHandler.
    """
    df = pd.read_csv(file, index_col=0)
    # Convert index from int to datetime object
    df.index = pd.to_datetime(df.index)
    dh = DataHandler(df)
    name = os.path.basename(file)
    return (name, dh)

def get_csvs_in_dir(folder_path):
    """
    Gets the csvs in a directory.
    Parameters:
    - folder_path: Directory containing the csvs.
    Returns:
    - An array of the csv file paths.
    """
    csvs = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            if filename.endswith(".csv"):
                csvs.append(file_path)
    return csvs


def run_job(data_result, track_times):
    """
    Processes a single unit of data using DataHandler.
    Parameters:
    - data_result: Tuple of the file name and its corresponding DataHandler.
    - track_times: Boolean to determine whether run times are added to the
                   output.
    Returns:
    - A dictionary containing the name of the data and the processed report.
    If there was an error with processing, only the name of the data is
    returned.
    """
    name = data_result[0]
    data_handler = data_result[1]

    try:
        data_handler.run_pipeline()
        report = data_handler.report(verbose=False, return_values=True)
        report["name"] = name
        if track_times:
            report["total_time"] = data_handler.total_time
    except:
        report = {}
        report["name"] = name
    return report

def remote_run_job(data_result, track_times):
    """
    Processes a single unit of data using DataHandler.
    Parameters:
    - data_result: Tuple of the file name and its corresponding DataHandler.
    - track_times: Boolean to determine whether run times are added to the 
                   output.
    Returns:
    - A dictionary containing the name of the data and the processed report.
    If there was an error with processing, only the name of the data is 
    returned.
    """
    name = data_result[0]
    data_handler = data_result[1]
    column = data_result[2]
    
    try:    
        data_handler.run_pipeline(power_col=column)
        report = data_handler.report(verbose=False, return_values=True)
        report["name"] = name
        if track_times:
            report["total_time"] = data_handler.total_time
    except:
        report = {}
        report["name"] = name
    return report


def run_pipelines(data_list, data_retrieval_fn, track_times=True):
    """
    Executes the data processing pipelines on the provided data list.
    Parameters:
    - data_list: List of the data files to process
    - data_retrieval_fn: Function that takes an item from the data list and
    returns a tuple of the name of the data item and it's DataHandler.
    - track_times: Boolean to determine whether run times are added to the
                   output.
    Returns:
    - A list of reports generated from processing each data entry.
    """
    reports = []

    for filename in data_list:
        data_result = delayed(data_retrieval_fn)(filename)
        result = delayed(run_job)(data_result, track_times)
        reports.append(result)

    with performance_report(filename="dask-report.html"):
        reports = compute(
            *reports,
        )

    return reports


@click.command()
@click.option("--dir", help="Directory containing csvs of solar data.")
@click.option(
    "--db_list",
    help="File containing sites to be retrieved from the cassandra database.",
)
@click.option(
    "--n_workers", default=1, help="Number of parallel workers to process with."
)
@click.option("--n_threads", default=1, help="Number of threads per worker.")
@click.option(
    "--mem_limit",
    default="16GB",
    help="Maximum memory that each worker is able to use.",
)
@click.option(
    "--track_times",
    default=False,
    help="Tracks the total time to process each data set.",
)
@click.option("--worker_log_file", default="", help="Output to save worker logs to.")
@click.argument("output")
def cli(
    dir, db_list, n_workers, n_threads, mem_limit, track_times, worker_log_file, output
):
    cluster = LocalCluster(
        n_workers=n_workers, threads_per_worker=n_threads, memory_limit=mem_limit
    )

    client = Client(cluster)

    if dir:
        reports = run_pipelines(get_csvs_in_dir(dir), local_csv_to_dh, track_times)
    elif db_list:
        with tempfile.TemporaryDirectory() as temp_dir:
            with open(db_list, "r") as file:
                for site in file:
                    df = load_cassandra_data(site)
                    dh = DataHandler(df, convert_to_ts=True)
                    dh_keys = dh.keys
                    raw_df = dh.data_frame_raw
                    # Convert timestamp to integer
                    raw_df.index = raw_df.index.view("int")
                    # Split out systems to individual files and save to csv
                    for key in dh_keys:
                        site = key[0][0]
                        site = site.strip()
                        system = key[0][1]
                        system = system.strip()
                        column = key[1]
                        filename = f"{temp_dir}/{site}_{system}.csv"
                        series = raw_df[column]
                        series.to_csv(filename)
            reports = run_pipelines(
                get_csvs_in_dir(temp_dir), local_csv_to_dh, track_times
            )
    else:
        raise Exception("--dir or --db_list must be provided")

    df = pd.DataFrame(reports)
    df.to_csv(output)

    if worker_log_file != "":
        logs = client.get_worker_logs()
        with open(worker_log_file, "w") as fp:
            json.dump(logs, fp, indent=4)


def generate_task_local(filename, track_times=True):
    """
    Generate the analysis task for a given local file.

    Parameters:
    - filename: Name of the local file
    - track_times: Booleans to determine whether run times are added
    to the output

    Returns:
    - A Dask delayed task object for the data analysis, which depends
    on the ingest task.
    """
    task_ingest = delayed(local_csv_to_dh)(filename)
    task_analyze = delayed(run_job)(task_ingest, track_times)
    return task_analyze


def remote_site_to_dhs(site, track_times=True):
    """
    Converts a remote database site into a solar-data-tools DataHandler.
    Parameters:
    - site: remote site.
    Returns:
    - A tuple of the unique identifier and its corresponding DataHandler.
    """
    result = []
    df = load_cassandra_data(site)
    dh = DataHandler(df, convert_to_ts=True)
    dh.data_frame_raw.index = dh.data_frame_raw.index.view("int")
    dh_keys = dh.keys
    for key in dh_keys:
        site = key[0][0]
        site = site.strip()
        system = key[0][1]
        system = system.strip()
        name = site + system
        column = key[1]
        task_analyze = delayed(remote_run_job)((name, dh, column), track_times)
        result.append(task_analyze)
    return result


def generate_tasks_directory_local(directory, track_times=True):
    """
    Generate the analysis tasks for a given directory containing csv's.

    Parameters:
    - directory: Path of the directory containing csv's
    - track_times: Booleans to determine whether run times are added
    to the output

    Returns:
    - A list of Dask delayed task objects for the data analysis,
    each of which depends on an ingest task.
    """
    result = []
    for filename in get_csvs_in_dir(directory):
        result.append(generate_task_local(filename))
    return result


def generate_tasks_remote_database(db_list):
    """
    Generate the analysis tasks for remote database.

    Parameters:
    - db_list: Path of the directory containing a list of sites from remote database

    Returns:
    - A list of Dask delayed task objects for the data analysis,
    each of which depends on an ingest task.
    """
    result = []
    with open(db_list, "r") as file:
        for site in file:
            result.extend(remote_site_to_dhs(site))
    return result

def execute_tasks(task_list):
    """
    Execute a list of tasks.

    NOTE: The Dask cluster should be
    intialized before calling this function.

    Parameters:
    - task_list: A list of dask delayed object

    Returns:
    - A list of reports from execution
    """
    reports = compute(
        *task_list,
    )
    return reports


# TODO: we need functions to generate tasks from remote sources

if __name__ == "__main__":
    obj_list = generate_tasks_directory_local("./")

    # Note that dask.visualize works differently without
    # iPython notebook.
    obj_list[0].visualize()

    client = Client(threads_per_worker=4, n_workers=2)

    execute_tasks(obj_list)
