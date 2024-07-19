"""
Azure test script for Dask using S3Bucket data plug

It takes in the following arguments:
    -w:int      worker number     (default=4)
    -t:int      thread number     (default=2)
    -m:float    memory            (default=15.36)
    -v:bool     verbose           (default=False)
    -b:string   bucket name       (default='pvinsight-dask-baseline')
    -r:string   result path       (default='../results/')

Example command to run this test script:
    python rev_azu_base_dask.py -w 2 -t 2

Before running the script, make sure to set up the environment variables for Azure:
    RESOURCE_GROUP
    VNET
    SECURITY_GROUP
"""

import os, argparse
from time import strftime
from sdt_dask.clients.azure.azure_client import AzureClient
from sdt_dask.dataplugs.S3Bucket_plug import S3Bucket
from sdt_dask.dask_tool.runner import Runner

time_stamp = strftime("%Y%m%d-%H%M%S")

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
    default=15.63,
    help=(
        "Declare memory limit per worker. "
        "Example --memory 15.36', default=15.36"),
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


resource_group=os.getenv("RESOURCE_GROUP")
vnet=os.getenv("VNET")
security_group=os.getenv("SECURITY_GROUP")
image = "slacgismo/sdt-v1:latest"

WORKERS = int(options.workers)
THREADS = int(options.threads)
MEMORY = float(options.memory)
VERBOSE = bool(options.verbose)
BUCKET = options.bucket
RESULT_PATH = options.result_path
RESULT_DIR = f"{RESULT_PATH}/{WORKERS}workers-{THREADS}threads/"

# check whether result path exists, if not create it
os.makedirs(f"{RESULT_DIR}/", exist_ok=True)

print('Grabbing files from bucket: %s', BUCKET)

# Defined S3 Bucket dataplug
data_plug = S3Bucket(bucket_name=BUCKET)

# Required for S3 Bucket pull keys as a list given as output
key_list = data_plug._pull_keys()
KEYS = [(key,) for key in key_list]

# Sets the dask fargate client and dask tool
# Uses the dask tool for computation
if __name__ == '__main__':
    # Dask Fargate client Setup
    client_setup = AzureClient(workers=WORKERS,
                         threads=THREADS,
                         memory=MEMORY,
                         resource_group=resource_group,
                         vnet=vnet,
                         security_group=security_group,
                         docker_image=image,
                         location="westus2",
                         vm_size="Standard_D4s_v3",
                         public_ingress=True,
                         disk_size=30,
                         # Environment variables needed to let VM have access to the S3 bucket
                         env_vars={ 
                            "AWS_ACCESS_KEY_ID": os.getenv["AWS_ACCESS_KEY_ID"],
                            "AWS_SECRET_ACCESS_KEY": os.getenv["AWS_SECRET_ACCESS_KEY"],
                            "AWS_REGION": os.getenv["AWS_REGION"]
                        }
                        )
    # Dask Local Client Initialization
    client = client_setup.init_client()

    # Dask Tool initialization and set up
    dask_tool = Runner(client=client,
                       output_path=f"{RESULT_DIR}")
    dask_tool.set_up(KEYS, data_plug=data_plug, fix_shifts=True, verbose=VERBOSE)

    # Dask Tool Task Compute
    dask_tool.get_result()

