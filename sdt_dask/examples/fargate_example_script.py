# Need to check python path variables
import sys

# Appending path to successfully import solar data tool modules
sys.path.append('..\\..')
sys.path.append('..\\..\\..')

import glob, os
import pandas as pd
from dask import delayed
from dask.distributed import performance_report
from solardatatools import DataHandler
from sdt_dask.clients.aws.fargate import Fargate
from sdt_dask.dataplugs.pvdaq_plug import PVDAQPlug

# The Tag, VPC, image, workers, threads per worker and environment need to be user defined and passed to the client class
PA_NUMBER = os.getenv("project-pa-number")
TAGS = {
    "project-pa-number": PA_NUMBER,
    "project": "pvinsight"
}
VPC = "vpc-ab2ff6d3" # for us-west-2
IMAGE = "nimishy/sdt-dask-windows:latest"

AWS_DEFAULT_REGION = os.getenv('AWS_DEFAULT_REGION')
ENVIRONMENT = {
    'AWS_ACCESS_KEY_ID' : os.getenv('AWS_ACCESS_KEY_ID'),
    'AWS_SECRET_ACCESS_KEY' : os.getenv('AWS_SECRET_ACCESS_KEY')
}

WORKERS = 3
THREADS_PER_WORKER = 1


# Instantiate a data plug
data_plug = PVDAQPlug()
KEYS = [(34, 2011), (35, 2015), (51,2012)] # site ID and year pairs

# Define the pipeline run for as single dataset
def run_pipeline(datahandler, solver, solver_convex, verbose=False):
    # Need to call this separately to have it run correctly in task graph 
    # since it doesn't return anything
    datahandler.run_pipeline(solver=solver, solver_convex=solver_convex, verbose=verbose)
    return datahandler


# Call above functions in a for loop over the keys
# and collect results in a DataFrame
reports = []
runtimes = []

for key in KEYS:
    
    df = delayed(data_plug.get_data)(key)
    dh = delayed(DataHandler)(df)
    dh_run = delayed(run_pipeline)(dh, solver="OSQP", solver_convex="OSQP", verbose=True)
    report = dh_run.report
    runtime = dh_run.total_time
    
    report = delayed(report)(return_values=True, verbose=False)
    runtime = delayed(runtime)
    
    reports.append(report)
    runtimes.append(runtime)


df_reports = delayed(pd.DataFrame)(reports)
df_reports = delayed(df_reports.assign)(runtime=runtimes, keys=KEYS)

# Visualizing the graph
df_reports.visualize()

# Instantiate Fargate cluster and dask client
client = Fargate().init_client(image=IMAGE, 
                               tags=TAGS, 
                               vpc=VPC, 
                               region_name=AWS_DEFAULT_REGION,
                               environment=ENVIRONMENT,
                               n_workers=WORKERS,
                               threads_per_worker=THREADS_PER_WORKER
                               )

# Compute tasks on cluster and save results
with performance_report(filename="../results/dask-report-fargate-or.html"):
    summary_table = client.compute(df_reports)
    df = summary_table.result()
    df.to_csv('summary_report_fargate.csv')

client.shutdown()
