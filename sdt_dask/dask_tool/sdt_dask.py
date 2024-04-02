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

def capture_exception_to_attribute_runpipeline(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            for arg in args:
                if isinstance(arg, DataHandler):
                    arg.run_pipeline_error = str(e)
                    break
            print(f"Error captured at run pipeline and stored in DataHandler: {e}\n")
            return arg
    return wrapper

@capture_exception_to_attribute_runpipeline
def run_pipeline(datahandler, **kwargs):
    datahandler.run_pipeline(**kwargs)
    return datahandler

def capture_exception_to_attribute_runlossanalysis(func):
    def wrapper(*args):
        try:
            return func(*args)
        except Exception as e:
            for arg in args:
                if isinstance(arg, DataHandler):
                    arg.run_loss_analysis_error = str(e)
                    break
            print(f"Error captured at loss analysis and stored in DataHandler: {e}\n")
            return arg
    return wrapper

@capture_exception_to_attribute_runlossanalysis
def run_loss_analysis(datahandler):
    datahandler.run_loss_factor_analysis()
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
        losses = []
        run_pipeline_errors = []
        run_loss_analysis_errors = []
        run_pipeline_report_errors = []
        loss_analysis_report_errors = []

        class Data:
            def __init__(self, report, loss_report, runtime, run_pipeline_errors, run_loss_analysis_errors, run_pipeline_report_errors, loss_analysis_report_errors):
                self.report = report
                self.loss_report = loss_report
                self.runtime = runtime
                self.run_pipeline_errors = run_pipeline_errors
                self.run_loss_analysis_errors = run_loss_analysis_errors
                self.run_pipeline_report_errors = run_pipeline_report_errors
                self.loss_analysis_report_errors = loss_analysis_report_errors

        def helper(datahandler, key):
            report = None
            loss_report = None
            runtime = None
            run_pipeline_error = None
            run_loss_analysis_error = None
            run_pipeline_report_error = None
            loss_analysis_report_error = None
            try:
                report = datahandler.report(return_values=True, verbose=False)
            except Exception as e:
                datahandler.run_pipeline_report_error = str(e)

            try:
                loss_report = datahandler.loss_analysis.report()
            except Exception as e:
                datahandler.loss_analysis_report_error = str(e)

            try:
                runtime = datahandler.total_time
            except Exception as e:
                print(e)
            
            if hasattr(datahandler, "run_pipeline_error"):
                run_pipeline_error = datahandler.run_pipeline_error

            if hasattr(datahandler, "run_loss_analysis_error"):
                run_loss_analysis_error = datahandler.run_loss_analysis_error

            if hasattr(datahandler, "run_pipeline_report_error"):
                run_pipeline_report_error = datahandler.run_pipeline_report_error

            if hasattr(datahandler, "loss_analysis_report_error"):
                loss_analysis_report_error = datahandler.loss_analysis_report_error
            
            return Data(report, loss_report, runtime, run_pipeline_error, run_loss_analysis_error, run_pipeline_report_error, loss_analysis_report_error)

        def helper_data(datas):
            return [data if data is not None else {} for data in datas]
        
        def helper_error(datas):
            return [data if data is not None else "No Error" for data in datas]

        for key in KEYS:
            # TODO: to check if a key is valid explicitly

            df = delayed(self.data_plug.get_data)(key)
            dh = delayed(DataHandler)(df)
            dh_run = delayed(run_pipeline)(dh, **kwargs)
            dh_run = delayed(run_loss_analysis)(dh_run)
        
            data = delayed(helper)(dh_run, key)

            reports.append(data.report)
            losses.append(data.loss_report)
            runtimes.append(data.runtime)
            run_pipeline_errors.append(data.run_pipeline_errors)
            run_loss_analysis_errors.append(data.run_loss_analysis_errors)
            run_pipeline_report_errors.append(data.run_pipeline_report_errors)
            loss_analysis_report_errors.append(data.loss_analysis_report_errors)
    
        reports = delayed(helper_data)(reports)
        losses = delayed(helper_data)(losses)
        run_pipeline_errors = delayed(helper_error)(run_pipeline_errors)
        run_loss_analysis_errors = delayed(helper_error)(run_loss_analysis_errors)
        run_pipeline_report_errors = delayed(helper_error)(run_pipeline_report_errors)
        loss_analysis_report_errors = delayed(helper_error)(loss_analysis_report_errors)
        self.df_reports = delayed(pd.DataFrame)(reports)
        self.loss_reports = delayed(pd.DataFrame)(losses)
        # append losses to the report
        self.df_reports = delayed(pd.concat)([self.df_reports, self.loss_reports], axis=1)
        self.df_reports = delayed(self.df_reports.assign)(runtime=runtimes, key=KEYS)
        self.df_reports = delayed(self.df_reports.assign)(run_pipeline_error=run_pipeline_errors, run_loss_analysis_error=run_loss_analysis_errors, run_pipeline_report_error=run_pipeline_report_errors, loss_analysis_report_error=loss_analysis_report_errors)

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