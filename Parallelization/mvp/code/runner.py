from dask.distributed import Client
from dask_cloudprovider.aws import FargateCluster
from solardatatools import DataHandler
import boto3
import pandas as pd
from dask import delayed, compute
from dask.distributed import performance_report, LocalCluster
import os


def s3_csv_to_dh(name):
    """
    Converts a s3 CSV file into a solar-data-tools DataHandler.
    Parameters:
    - file: Path to the CSV file.
    Returns:
    - A tuple of the file name and its corresponding DataHandler.
    """
    df = pd.read_csv(name, index_col=0)
    # Convert index from int to datetime object
    df.index = pd.to_datetime(df.index)
    dh = DataHandler(df)
    # name = name.split('/')[-1].removesuffix('.csv')
    name = name.removesuffix('.csv')
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
    csvs = []
    data_bucket = s3.Bucket(bucket)
    for object_summary in data_bucket.objects.filter(Prefix=prefix):
        if object_summary.key.endswith('.csv'):
            file = f"s3://{bucket}/{object_summary.key}"
            csvs.append(file)
    return csvs

def run_job(data_result, track_times):
    """
    Processes a single unit of data using DataHandler.
    Parameters:
    - data: The input data to be processed.
    - data_retrieval_fn: Function to retrieve and format format each data entry.
                         Should return a tuple with the name of the site and a
                         pandas data_frame of the solar data.
    Returns:
    - A dictionary containing the name of the data and the processed report.
    """
    name = data_result[0]
    data_handler = data_result[1]
    try:    
        data_handler.run_pipeline()
        report = data_handler.report(verbose=False, return_values=True)
        report["name"] = name
        if track_times:
            report["total_time"] = data_handler.total_time
    except Exception as e:
        print(e)
        report = {}
        report["name"] = name
    return report

def run_pipelines(data_list, data_retrieval_fn, track_times = True):
    """
    Executes the data processing pipelines on the provided data list.
    Returns:
    - A list of reports generated from processing each data entry.
    """
    reports = []

    for filename in data_list:
        data_result = delayed(data_retrieval_fn)(filename)
        result = delayed(run_job)(data_result, track_times)
        reports.append(result)
    
    with performance_report(filename="dask-report-fargate.html"):
        reports = compute(*reports, )
    
    return reports

if __name__ == "__main__":
    # Pull credentials to pass to ECS for s3 access
    aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY')

    # Ensure credentials are available
    if not aws_access_key_id or not aws_secret_access_key:
        raise ValueError("AWS credentials not found in environment variables")

    tags = {"project-pa-number": "21691-H2001", "project": "pvinsight"}
    aws_access_key_id = os.environ['AWS_ACCESS_KEY_ID']
    aws_secret_access_key = os.environ['AWS_SECRET_ACCESS_KEY']

    cluster = FargateCluster(
        tags=tags,
        image="jjcrush/dask:latest",
        vpc="vpc-ab2ff6d3",
        region_name="us-west-2",
        n_workers=10,
        worker_nthreads=1,
        environment={
            'AWS_ACCESS_KEY_ID': aws_access_key_id,
            'AWS_SECRET_ACCESS_KEY': aws_secret_access_key
        },
    )

    # cluster = LocalCluster(n_workers=16, 
    #         threads_per_worker=1,
    #         memory_limit='16GB')

    client = Client(cluster)
    s3 = boto3.resource('s3')
    reports = run_pipelines(
        get_csvs_in_s3(
            s3,
            'pv.insight.sdtdataset',
            'cassandra_data_extract'
        ),
        s3_csv_to_dh,
        True)

    df = pd.DataFrame(reports)
    df.to_csv('processing_report_fargate.csv')
