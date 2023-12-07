from dask.distributed import Client
from dask_yarn import YarnCluster

from solardatatools.dataio import load_cassandra_data
from solardatatools import DataHandler
from statistical_clear_sky import SCSF

import dask

cluster = YarnCluster(worker_memory='16GiB')
client = Client(cluster)

@dask.delayed
def pull_and_run(site_id):
    df = load_cassandra_data(site_id, cluster_ip="54.176.95.208")
    dh = DataHandler(df, convert_to_ts=True)
    dh.run_pipeline(power_col='ac_power_01')
    return dh.report(return_values=True)

results = []
site_ids = ["TAAJ01021775", "001C4B0008A5", "TABG01081601"]
for si in site_ids:
    results.append(pull_and_run(si))
dask.compute(results)

print(results)