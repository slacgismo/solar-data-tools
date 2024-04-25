import os
import pandas as pd
from dask import delayed
from dask.distributed import Client, performance_report
from solardatatools import DataHandler
from dask_cloudprovider.aws import FargateCluster
from sdt_dask.dataplugs.pvdaq_plug import PVDAQPlug


# Set up
PA_NUMBER = os.getenv("project-pa-number")
TAGS = {
    "project-pa-number": PA_NUMBER,
    "project": "pvinsight"
}

IMAGE = "smiskov/dask-sdt-sm-2:latest"

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION")
VPC = "vpc-ab2ff6d3" # for us-west-2


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

for key in KEYS:
    
    df = delayed(data_plug.get_data)(key)
    dh = delayed(DataHandler)(df)
    dh_run = delayed(run_pipeline)(dh, solver="OSQP", solver_convex="OSQP", verbose=True)
    report = dh_run.report
    report = delayed(report)(return_values=True, verbose=False)
    reports.append(report)

df_reports = delayed(pd.DataFrame)(reports)

# Visualize task graph
df_reports.visualize(filename='sdt_graph_or.png')

# Instantiate Fargate cluster
cluster = FargateCluster(
    tags=TAGS,
    image=IMAGE,
    vpc=VPC,
    region_name=AWS_DEFAULT_REGION,
    n_workers=3,
    worker_nthreads=1,
    environment={
        'AWS_ACCESS_KEY_ID': AWS_ACCESS_KEY_ID,
        'AWS_SECRET_ACCESS_KEY': AWS_SECRET_ACCESS_KEY
    }
)

client = Client(cluster)
print(client.dashboard_link)

# Compute tasks on cluster and save results
with performance_report(filename="../results/dask-report-fargate-or.html"):
    summary_table = client.compute(df_reports)
    df = summary_table.result()
    df.to_csv('summary_report_fargate.csv')

client.shutdown()
