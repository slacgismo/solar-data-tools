"""

This module provides a class to run the SolarDataTools pipeline on a Dask cluster.
It takes a data plug and a dask client as input and runs the pipeline on the data plug

See the README and tool_demo_SDTDask.ipynb for more information

"""
import os
import pandas as pd
from dask import delayed
from dask.distributed import performance_report
from solardatatools import DataHandler

def run_pipeline(datahandler, **kwargs):
    """function to run the pipeline on a datahandler object

    user can pass any keyword arguments to the pipeline

    Args:
        datahandler (:obj:`DataHandler`): The datahandler object.
        **kwargs: Optional parameters.

    Returns:
        DataHandler: The datahandler object after running the pipeline.
    """
    # TODO: add loss analysis
    # TODO: if dataset failed to run, throw python error
    datahandler.run_pipeline(**kwargs)
    return datahandler

class SDTDask:
    """A class to run the SolarDataTools pipeline on a Dask cluster.

    Will handle invalid data keys and failed datasets.

    Attributes:
        data_plug (:obj:`DataPlug`): The data plug object.
        client (:obj:`Client`): The Dask client object.
        output_path (str): The path to save the results.

    """

    def __init__(self, data_plug, client, output_path="../results/"):
        self.data_plug = data_plug
        self.client = client
        self.output_path = output_path
        
    def set_up(self, KEYS, **kwargs):
        """function to set up the pipeline on the data plug

        Call run_pipeline functions in a for loop over the keys
        and collect results in a DataFrame

        Args:
            KEYS (list): List of tuples
            **kwargs: Optional parameters.

        """

        reports = []
        runtimes = []

        for key in KEYS:
            # TODO: to check if a key is valid explicitly

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

    def visualize(self, filename="sdt_graph.png"):
        # visualize the pipeline, user should have graphviz installed
        self.df_reports.visualize(filename)

    def get_result(self):
        self.get_report()

    def get_report(self):
        # test if the filepath exist, if not create it
        if not os.path.exists(self.output_path):
            print("output path does not exist, creating it...")
            os.makedirs(self.output_path)
        # Compute tasks on cluster and save results
        with performance_report(self.output_path + "/dask-report.html"):
            summary_table = self.client.compute(self.df_reports)
            df = summary_table.result()
            df.to_csv(self.output_path + "/summary_report.csv")

        self.client.shutdown()