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
    try:
        datahandler.run_pipeline(**kwargs)
        datahandler.run_loss_factor_analysis()
        return datahandler
    except Exception as e:
        print(f"Error running pipeline: {e}")
        return None
    

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
        losses = []

        class Data:
            def __init__(self, report, loss_report, runtime):
                self.report = report
                self.loss_report = loss_report
                self.runtime = runtime

        def helper(datahandler, key):
            report = None
            loss_report = None
            runtime = None
            try:
                report = datahandler.report(return_values=True, verbose=False)
                loss_report = datahandler.loss_analysis.report()
                runtime = datahandler.total_time
            except Exception as e:
                print(e)
            
            return Data(report, loss_report, runtime)

        def helper_data(datas):
            return [data if data is not None else {} for data in datas]

        for key in KEYS:
            # TODO: to check if a key is valid explicitly

            df = delayed(self.data_plug.get_data)(key)
            dh = delayed(DataHandler)(df)
            dh_run = delayed(run_pipeline)(dh, **kwargs)
        
            data = delayed(helper)(dh_run, key)

            reports.append(data.report)
            losses.append(data.loss_report)
            runtimes.append(data.runtime)
    
        reports = delayed(helper_data)(reports)
        losses = delayed(helper_data)(losses)
        self.df_reports = delayed(pd.DataFrame)(reports)
        self.loss_reports = delayed(pd.DataFrame)(losses)
        # append losses to the report
        self.df_reports = delayed(pd.concat)([self.df_reports, self.loss_reports], axis=1)
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