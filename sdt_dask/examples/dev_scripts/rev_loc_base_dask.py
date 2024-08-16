"""
Local test script for Dask using local data plug

It takes in the following arguments:
    -w:int      worker number     (default=4)
    -t:int      thread number     (default=2)
    -m:int      memory            (default=5)
    -v:bool     verbose           (default=False)
    -f:string   file path         (default=None)
    -r:string   result path       (default='../results/')

Example command to run this test script:
    python rev_loc_base_dask.py -w 2 -t 2 -m 6 -f ../example_data/ -p ../results/
"""
import glob, os, argparse
from sdt_dask.dataplugs.csv_plug import LocalFiles
from sdt_dask.clients.local_client import LocalClient
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
    "-f",
    "--file_path",
    default=None,
    help=(
        "Provide the directory path for file computation. "
        "Example --file ../results/', default=None"),
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

WORKERS = int(options.workers)
THREADS = int(options.threads)
MEMORY = float(options.memory)
VERBOSE = bool(options.verbose)
FILE_PATH = options.file_path
RESULT_PATH = options.result_path
RESULT_DIR = f"{RESULT_PATH}/{WORKERS}workers-{THREADS}threads/"

# Defined Local CSV dataplug
data_plug = LocalFiles(path_to_files=FILE_PATH)
KEYS = [(os.path.basename(fname)[:-4],) for fname in glob.glob(FILE_PATH + "*")]

os.makedirs(RESULT_DIR, exist_ok=True)

# Setup the dask local client and dask tool
# Uses the dask tool for computation
# The conditional statement is required to allow
# process spawning
if __name__ == '__main__':
    # Dask Local client Setup
    client_setup = LocalClient(workers=WORKERS,
                               threads=THREADS,
                               memory=MEMORY)
    # Dask Local Client Initialization
    client = client_setup.init_client()

    # Dask Tool initialization and set up
    dask_tool = Runner(client=client, output_path=RESULT_DIR)
    dask_tool.set_up(KEYS, data_plug=data_plug, fix_shifts=True, verbose=VERBOSE)

    # Dask Tool Task Compute and client shutdown
    dask_tool.get_result()