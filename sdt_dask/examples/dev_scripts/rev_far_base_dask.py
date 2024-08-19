"""
AWS test script for Dask using S3Bucket data plug

It takes in the following arguments:
    -w:int      worker number     (default=4)
    -t:int      thread number     (default=2)
    -m:int      memory            (default=16)
    -v:bool     verbose           (default=False)
    -b:string   bucket name       (default='pvinsight-dask-baseline')
    -r:string   result path       (default='../results/')

Example command to run this test script:
    python rev_far_base_dask.py -w 2 -t 2

Before running the script, make sure to set up the environment variables:
    project_pa_number
    AWS_DEFAULT_REGION
    AWS_ACCESS_KEY_ID
    AWS_SECRET_ACCESS_KEY
"""
import os, argparse, sys
from time import strftime
from sdt_dask.dataplugs.S3Bucket_plug import S3Bucket
from sdt_dask.clients.aws.fargate_client import FargateClient
from sdt_dask.dask_tool.runner import Runner

parser = argparse.ArgumentParser()

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
    default=16,
    help=(
        "Declare memory limit per worker. from 8 to 30 for 4vCPU "
        "Example --memory 8, default=8"),
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


WORKERS = int(options.workers)
THREADS = int(options.threads)
MEMORY = int(options.memory)
VERBOSE = bool(options.verbose)
BUCKET = options.bucket
RESULT_PATH = options.result_path
RESULT_DIR = f"{RESULT_PATH}/{WORKERS}workers-{THREADS}threads/"

# check whether result path exists, if not create it
os.makedirs(RESULT_DIR, exist_ok=True)
def check_enviornment_variables():
    if "project_pa_number" in os.environ:
        PA_NUMBER = os.getenv("project_pa_number")
        TAGS = {
            "project-pa-number": PA_NUMBER,
            "project": "pvinsight"
        }
    else:
        print("Please set the environment variable project_pa_number")
        sys.exit(0)
    if "AWS_DEFAULT_REGION" in os.environ:
        AWS_DEFAULT_REGION = os.getenv('AWS_DEFAULT_REGION')
    else:
        print("Please set the environment variable AWS_DEFAULT_REGION")
        sys.exit(0)

    if "AWS_ACCESS_KEY_ID" in os.environ and "AWS_SECRET_ACCESS_KEY" in os.environ:
        ENVIRONMENT = {
            'AWS_ACCESS_KEY_ID': os.getenv('AWS_ACCESS_KEY_ID'),
            'AWS_SECRET_ACCESS_KEY': os.getenv('AWS_SECRET_ACCESS_KEY')
        }
    else:
        print("Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables")
        sys.exit(0)

    return TAGS, ENVIRONMENT, AWS_DEFAULT_REGION

VPC = "vpc-ab2ff6d3"  # for us-west-2

IMAGE = "slacgismo/sdt-v1:latest"



# Sets the dask fargate client and dask tool
# Uses the dask tool for computation
if __name__ == '__main__':
    TAGS, ENVIRONMENT, AWS_DEFAULT_REGION = check_enviornment_variables()

    # Defined S3 Bucket dataplug
    data_plug = S3Bucket(bucket_name=BUCKET)

    # Required for S3 Bucket pull keys as a list given as output
    key_list = data_plug._pull_keys()
    KEYS = [(key,) for key in key_list]

    # Dask Fargate client Setup
    client_setup = FargateClient(workers=WORKERS,
                                 threads=THREADS,
                                 memory=MEMORY,
                                 image=IMAGE,
                                 tags=TAGS,
                                 vpc=VPC,
                                 region_name=AWS_DEFAULT_REGION,
                                 environment=ENVIRONMENT)
    # Dask Local Client Initialization
    client = client_setup.init_client()

    # Dask Tool initialization and set up
    dask_tool = Runner(client=client, output_path=RESULT_DIR)
    dask_tool.set_up(KEYS, data_plug=data_plug, fix_shifts=True, verbose=VERBOSE)

    time_stamp = strftime("%Y%m%d-%H%M%S")

    # Dask Tool Task Compute and shutdown client
    output_html = f"new_rev_far_dask-report_{WORKERS}w-{THREADS}t-{time_stamp}.html"
    output_csv = f"new_rev_far_summary_report_{WORKERS}w-{THREADS}t-{time_stamp}.csv"
    dask_tool.get_result(dask_report = output_html, summary_report = output_csv)


