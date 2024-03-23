import os
import pandas as pd
from dask import delayed
from dask.distributed import performance_report
from solardatatools import DataHandler

# Define the pipeline run for as single dataset
def run_pipeline(datahandler, **kwargs):
    # Need to call this separately to have it run correctly in task graph 
    # since it doesn't return anything
    datahandler.run_pipeline(**kwargs)
    return datahandler

class SDTDask:

    def __init__(self, data_plug, client):
        self.data_plug = data_plug
        self.client = client
        
    def execute(self, KEYS, **kwargs):
        # Call above functions in a for loop over the keys
        # and collect results in a DataFrame
        reports = []
        runtimes = []

        # KEYS example: [(34, 2011), (35, 2015), (51,2012)] # site ID and year pairs
        for key in KEYS:
            # TODO: to check if a key is valid explicitly

            # TODO: if dataset failed to run, throw python error

            df = delayed(self.data_plug.get_data)(key)
            dh = delayed(DataHandler)(df)
            dh_run = delayed(run_pipeline)(dh, **kwargs)
        
            report = dh_run.report
            runtime = dh_run.total_time
        
            report = delayed(report)(return_values=True, verbose=False)
            runtime = delayed(runtime)

            reports.append(report)
            runtimes.append(runtime)
    
        self.df_reports = delayed(pd.DataFrame)(reports)
        self.df_reports = delayed(self.df_reports.assign)(runtime=runtimes, keys=KEYS)

        # Visualize task graph
        # self.df_reports.visualize(filename='sdt_graph_or.png')

        self.get_report()

    def get_report(self):
        # Compute tasks on cluster and save results
        with performance_report(filename="../results/dask-report.html"):
            summary_table = self.client.compute(self.df_reports)
            df = summary_table.result()
            df.to_csv('../results/summary_report.csv')

        self.client.shutdown()

